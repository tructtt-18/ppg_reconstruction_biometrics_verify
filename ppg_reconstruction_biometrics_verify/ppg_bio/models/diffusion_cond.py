from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

@dataclass
class DiffusionConfig:
    sig_len: int = 512
    num_classes: int = 0                  # MUST be set
    base_ch: int = 64
    ch_mults: Tuple[int, ...] = (1, 2, 4, 4)   # down channels = base_ch * mult
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (64, 32)  # apply attention at these lengths (after downsample)
    num_heads: int = 4
    dropout: float = 0.0

    # diffusion schedule
    T: int = 1000
    beta_schedule: str = "cosine"  # "cosine" or "linear"

    # loss
    min_snr_gamma: float = 5.0     # Min-SNR gamma for reweighting

    # EMA
    ema_decay: float = 0.999

def _cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    steps = T + 1
    x = np.linspace(0, T, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-8, 0.999).astype(np.float32)


def _linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> np.ndarray:
    return np.linspace(beta_start, beta_end, T, dtype=np.float32)


class DiffusionSchedule:
    def __init__(self, betas: np.ndarray):
        betas = betas.astype(np.float32)
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas, axis=0)

        self.betas = tf.constant(betas, dtype=tf.float32)               # (T,)
        self.alphas = tf.constant(alphas, dtype=tf.float32)             # (T,)
        self.alpha_bar = tf.constant(alpha_bar, dtype=tf.float32)       # (T,)

        self.sqrt_alpha_bar = tf.sqrt(self.alpha_bar)                   # (T,)
        self.sqrt_one_minus_alpha_bar = tf.sqrt(1.0 - self.alpha_bar)   # (T,)

    @classmethod
    def create(cls, T: int, schedule: str = "cosine") -> "DiffusionSchedule":
        schedule = schedule.lower().strip()
        if schedule == "cosine":
            betas = _cosine_beta_schedule(T)
        elif schedule == "linear":
            betas = _linear_beta_schedule(T)
        else:
            raise ValueError("beta_schedule must be 'cosine' or 'linear'")
        return cls(betas)


def _extract(coeffs_1d: tf.Tensor, t: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    # t: (B,), int32
    out = tf.gather(coeffs_1d, t)                 # (B,)
    return tf.reshape(out, [tf.shape(x)[0], 1, 1])


# ---------------------------------------------------------------------
# Embeddings + conditioning blocks
# ---------------------------------------------------------------------

class SinusoidalTimeEmbedding(layers.Layer):
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)

    def call(self, t: tf.Tensor) -> tf.Tensor:
        t = tf.cast(t, tf.float32)
        half = self.dim // 2
        freqs = tf.exp(
            -tf.math.log(10000.0) * tf.range(half, dtype=tf.float32) / tf.cast(half - 1, tf.float32)
        )
        args = t[:, None] * freqs[None, :]
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        if self.dim % 2 == 1:
            emb = tf.pad(emb, [[0, 0], [0, 1]])
        return emb


class FiLM(layers.Layer):
    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.to_gamma = layers.Dense(self.channels)
        self.to_beta = layers.Dense(self.channels)

    def call(self, h: tf.Tensor, cond: tf.Tensor) -> tf.Tensor:
        # h: (B, T, C), cond: (B, D)
        g = self.to_gamma(cond)[:, None, :]   # (B, 1, C)
        b = self.to_beta(cond)[:, None, :]    # (B, 1, C)
        return g * h + b

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels})
        return cfg


class SelfAttention1D(layers.Layer):
    def __init__(self, channels: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.norm = layers.LayerNormalization()
        self.attn = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.channels // self.num_heads)
        self.proj = layers.Dense(self.channels)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        h = self.norm(x)
        h = self.attn(h, h)
        h = self.proj(h)
        return x + h


class CrossAttention1D(layers.Layer):
    def __init__(self, channels: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.norm = layers.LayerNormalization()
        self.attn = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.channels // self.num_heads)
        self.proj = layers.Dense(self.channels)
        self.token_proj = layers.Dense(self.channels, use_bias=False)

    def call(self, x: tf.Tensor, token: tf.Tensor) -> tf.Tensor:
        h = self.norm(x)
        if len(token.shape) == 2:
            token = token[:, None, :]
        token = self.token_proj(token)  # (B,1,C)
        h2 = self.attn(h, token)
        h2 = self.proj(h2)
        return x + h2


class ResBlock1D(layers.Layer):
    def __init__(self, channels: int, dropout: float, use_attn: bool, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.use_attn = bool(use_attn)
        self.num_heads = int(num_heads)

        self.norm1 = layers.LayerNormalization()
        self.conv1 = layers.Conv1D(self.channels, 3, padding="same")
        self.norm2 = layers.LayerNormalization()
        self.conv2 = layers.Conv1D(self.channels, 3, padding="same")

        self.film = FiLM(self.channels)

        self.drop = layers.Dropout(self.dropout) if self.dropout and self.dropout > 0 else None

        self.skip = None  # set in build if needed

        self.self_attn = SelfAttention1D(self.channels, self.num_heads) if self.use_attn else None
        self.cross_attn = CrossAttention1D(self.channels, self.num_heads) if self.use_attn else None

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        if in_ch != self.channels:
            self.skip = layers.Conv1D(self.channels, 1, padding="same")
        super().build(input_shape)

    def call(self, x: tf.Tensor, cond: tf.Tensor) -> tf.Tensor:
        h = self.norm1(x)
        h = tf.nn.swish(h)
        h = self.conv1(h)

        # inject conditioning (time + id)
        h = self.film(h, cond)

        h = self.norm2(h)
        h = tf.nn.swish(h)
        if self.drop is not None:
            h = self.drop(h)
        h = self.conv2(h)

        s = self.skip(x) if self.skip is not None else x
        out = s + h

        if self.self_attn is not None:
            out = self.self_attn(out)
            out = self.cross_attn(out, cond)  # "identity token" = cond (lightweight)

        return out


def build_denoiser(cfg: DiffusionConfig) -> tf.keras.Model:
    assert cfg.num_classes > 0, "cfg.num_classes must be set"

    x_in = layers.Input(shape=(cfg.sig_len, 1), name="x_t")
    t_in = layers.Input(shape=(), dtype="int32", name="t")
    y_in = layers.Input(shape=(), dtype="int32", name="y")

    # time embedding
    t_emb = SinusoidalTimeEmbedding(cfg.base_ch * 4, name="t_sin")(t_in)
    t_emb = layers.Dense(cfg.base_ch * 4, activation="swish", name="t_mlp1")(t_emb)
    t_emb = layers.Dense(cfg.base_ch * 4, activation="swish", name="t_mlp2")(t_emb)

    # identity embedding
    y_emb = layers.Embedding(cfg.num_classes, cfg.base_ch * 4, name="y_emb")(y_in)
    y_emb = layers.Dense(cfg.base_ch * 4, activation="swish", name="y_proj")(y_emb)

    # combined conditioning vector
    cond = layers.Add(name="cond_add")([t_emb, y_emb])  # (B, D)

    # input conv
    h = layers.Conv1D(cfg.base_ch, 3, padding="same", name="in_conv")(x_in)

    # Down path
    skips = []
    length = cfg.sig_len
    ch = cfg.base_ch
    for level, mult in enumerate(cfg.ch_mults):
        ch_out = cfg.base_ch * mult
        use_attn = length in cfg.attn_resolutions

        for b in range(cfg.num_res_blocks):
            h = ResBlock1D(ch_out, cfg.dropout, use_attn, cfg.num_heads, name=f"down{level}_res{b}")(h, cond)
            ch = ch_out
            skips.append(h)

        if level != len(cfg.ch_mults) - 1:
            # downsample
            h = layers.Conv1D(ch, 4, strides=2, padding="same", name=f"down{level}_ds")(h)
            length //= 2
            skips.append(h)

    # Middle
    use_attn = True
    h = ResBlock1D(ch, cfg.dropout, use_attn, cfg.num_heads, name="mid_res1")(h, cond)
    h = ResBlock1D(ch, cfg.dropout, use_attn, cfg.num_heads, name="mid_res2")(h, cond)

    # Up path (reverse)
    for level, mult in reversed(list(enumerate(cfg.ch_mults))):
        ch_out = cfg.base_ch * mult
        use_attn = length in cfg.attn_resolutions

        for b in range(cfg.num_res_blocks + 1):  # +1 to consume extra skip from downsample
            if skips:
                h = layers.Concatenate(name=f"up{level}_cat{b}")([h, skips.pop()])
            h = ResBlock1D(ch_out, cfg.dropout, use_attn, cfg.num_heads, name=f"up{level}_res{b}")(h, cond)

        if level != 0:
            # upsample
            h = layers.UpSampling1D(size=2, name=f"up{level}_us")(h)
            h = layers.Conv1D(ch_out, 3, padding="same", name=f"up{level}_conv")(h)
            length *= 2

    # output head
    h = layers.LayerNormalization(name="out_norm")(h)
    h = tf.nn.swish(h)
    v_out = layers.Conv1D(1, 3, padding="same", name="v_out")(h)

    return models.Model([x_in, t_in, y_in], v_out, name="diffusion_denoiser_v")

def q_sample(
    sched: DiffusionSchedule,
    x0: tf.Tensor,
    t: tf.Tensor,
    eps: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    if eps is None:
        eps = tf.random.normal(tf.shape(x0), dtype=tf.float32)
    a = _extract(sched.sqrt_alpha_bar, t, x0)
    b = _extract(sched.sqrt_one_minus_alpha_bar, t, x0)
    x_t = a * x0 + b * eps
    return x_t, eps


def v_target(
    sched: DiffusionSchedule,
    x0: tf.Tensor,
    eps: tf.Tensor,
    t: tf.Tensor,
) -> tf.Tensor:
    a = _extract(sched.sqrt_alpha_bar, t, x0)
    b = _extract(sched.sqrt_one_minus_alpha_bar, t, x0)
    return a * eps - b * x0


def reconstruct_x0_from_v(
    sched: DiffusionSchedule,
    x_t: tf.Tensor,
    v_pred: tf.Tensor,
    t: tf.Tensor,
) -> tf.Tensor:
    a = _extract(sched.sqrt_alpha_bar, t, x_t)
    b = _extract(sched.sqrt_one_minus_alpha_bar, t, x_t)
    return a * x_t - b * v_pred


def reconstruction_error(
    x: np.ndarray | tf.Tensor,
    x_hat: np.ndarray | tf.Tensor,
    p: int = 1,
    reduce_axis: Tuple[int, ...] = (1, 2),
) -> tf.Tensor:
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    xh_tf = tf.convert_to_tensor(x_hat, dtype=tf.float32)
    diff = x_tf - xh_tf
    if p == 1:
        return tf.reduce_mean(tf.abs(diff), axis=reduce_axis)
    if p == 2:
        return tf.sqrt(tf.reduce_mean(tf.square(diff), axis=reduce_axis) + 1e-12)
    raise ValueError("p must be 1 or 2")


# ---------------------------------------------------------------------
# Min-SNR weighting + loss helper
# ---------------------------------------------------------------------

def min_snr_weight(sched: DiffusionSchedule, t: tf.Tensor, gamma: float) -> tf.Tensor:
    ab = tf.gather(sched.alpha_bar, t)  # (B,)
    snr = ab / (1.0 - ab + 1e-12)
    w = tf.minimum(snr, gamma) / (snr + 1e-12)
    return tf.reshape(w, [tf.shape(t)[0], 1, 1])


def diffusion_v_loss(
    denoiser: tf.keras.Model,
    sched: DiffusionSchedule,
    x0: tf.Tensor,
    y: tf.Tensor,
    t: tf.Tensor,
    *,
    gamma: float = 5.0,
    training: bool = True,
) -> tf.Tensor:
    x_t, eps = q_sample(sched, x0, t, eps=None)
    v_t = v_target(sched, x0, eps, t)
    v_pred = denoiser([x_t, t, y], training=training)

    w = min_snr_weight(sched, t, gamma)
    loss = tf.reduce_mean(w * tf.square(v_pred - v_t))
    return loss


class EMA:
    def __init__(self, model: tf.keras.Model, decay: float = 0.999):
        self.decay = float(decay)
        self.ema_model = tf.keras.models.clone_model(model)
        self.ema_model.set_weights(model.get_weights())

    def update(self, model: tf.keras.Model):
        w = model.get_weights()
        w_ema = self.ema_model.get_weights()
        d = self.decay
        self.ema_model.set_weights([d * we + (1.0 - d) * wi for we, wi in zip(w_ema, w)])

@tf.function
def reconstruct_once(
    denoiser: tf.keras.Model,
    sched: DiffusionSchedule,
    x0: tf.Tensor,
    y: tf.Tensor,
    t: tf.Tensor,
) -> tf.Tensor:
    x_t, _ = q_sample(sched, x0, t, eps=None)
    v_pred = denoiser([x_t, t, y], training=False)
    x0_hat = reconstruct_x0_from_v(sched, x_t, v_pred, t)
    return x0_hat


def compute_re_matrix(
    x: np.ndarray,
    denoiser: tf.keras.Model,
    sched: DiffusionSchedule,
    num_classes: int,
    *,
    t_eval: int,
    p: int = 1,
) -> np.ndarray:
    x_tf = tf.convert_to_tensor(x, tf.float32)
    N = int(x_tf.shape[0])
    K = int(num_classes)
    t = tf.fill([N], tf.cast(t_eval, tf.int32))
    out = np.zeros((N, K), dtype=np.float32)

    for c in range(K):
        y_c = tf.fill([N], tf.cast(c, tf.int32))
        x_hat = reconstruct_once(denoiser, sched, x_tf, y_c, t)
        out[:, c] = reconstruction_error(x_tf, x_hat, p=p).numpy()

    return out


def predict_closed_set(re_mat: np.ndarray) -> np.ndarray:
    return np.argmin(re_mat, axis=1).astype(np.int64)


def accept_open_set(re_claim: np.ndarray, tau: float) -> np.ndarray:
    return (re_claim <= float(tau)).astype(np.int64)