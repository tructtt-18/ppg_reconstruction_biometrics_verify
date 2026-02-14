from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


@dataclass
class CGANConfig:
    sig_len: int = 512
    num_classes: int = 0  # must be set
    emb_dim: int = 64
    z_dim: int = 128
    g_base_ch: int = 256
    # GP / EMA
    gp_lambda: float = 10.0
    drift: float = 1e-3
    ema_decay: float = 0.999
    # Optional PBGAN++ feature loss



def G_block(h, ch: int, up: bool = True):
    if up:
        h = layers.UpSampling1D(2)(h)
    h = layers.Conv1D(ch, 5, padding="same")(h)
    h = layers.LayerNormalization()(h)
    h = layers.Activation("swish")(h)
    h = layers.GaussianNoise(0.02)(h)
    return h


def build_generator(cfg: CGANConfig) -> tf.keras.Model:
    assert cfg.num_classes > 0, "cfg.num_classes must be set"

    z_in = layers.Input(shape=(cfg.z_dim,), name="z")
    y_in = layers.Input(shape=(), dtype="int32", name="y")

    y = layers.Embedding(input_dim=cfg.num_classes, output_dim=cfg.emb_dim, name="y_emb")(y_in)
    y = layers.Dense(cfg.emb_dim, activation="swish", name="y_proj")(y)

    # fuse latent + condition
    h = layers.Concatenate(name="zy_concat")([z_in, y])
    h = layers.Dense((cfg.sig_len // 64) * cfg.g_base_ch, activation="swish", name="G_fc")(h)
    h = layers.Reshape((cfg.sig_len // 64, cfg.g_base_ch), name="G_reshape")(h)  # e.g., 512//64 = 8

    # upsample blocks: 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512
    h = G_block(h, 256)  # 16
    h = G_block(h, 192)  # 32
    h = G_block(h, 128)  # 64
    h = G_block(h, 96)   # 128
    h = G_block(h, 64)   # 256
    h = G_block(h, 32)   # 512

    x = layers.Conv1D(filters=1, kernel_size=7, padding="same", activation="tanh", name="G_out")(h)
    return models.Model(inputs=[z_in, y_in], outputs=x, name="G")



class SNDense(layers.Layer):
    def __init__(self, units, use_bias=True, power_iterations=1, **kw):
        super().__init__(**kw)
        self.units = int(units)
        self.use_bias = bool(use_bias)
        self.power_iterations = int(power_iterations)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="w",
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="b",
            )
        else:
            self.b = None

        self.u = self.add_weight(
            shape=(1, self.units),
            initializer=tf.random_normal_initializer(),
            trainable=False,
            name="u",
        )
        super().build(input_shape)

    def call(self, x):
        w_mat = self.w
        u = self.u
        for _ in range(self.power_iterations):
            v = tf.linalg.l2_normalize(tf.matmul(u, tf.transpose(w_mat)))
            u = tf.linalg.l2_normalize(tf.matmul(v, w_mat))
        self.u.assign(u)

        sigma = tf.matmul(tf.matmul(v, w_mat), tf.transpose(u))
        w_sn = w_mat / sigma
        y = tf.matmul(x, w_sn)
        if self.b is not None:
            y = y + self.b
        return y


class SNConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="same",
                 use_bias=True, power_iterations=1, **kw):
        super().__init__(**kw)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.padding = str(padding).upper()
        self.use_bias = bool(use_bias)
        self.power_iterations = int(power_iterations)

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        self.w = self.add_weight(
            shape=(self.kernel_size, in_ch, self.filters),
            initializer="glorot_uniform",
            trainable=True,
            name="w",
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
                name="b",
            )
        else:
            self.b = None

        self.u = self.add_weight(
            shape=(1, self.filters),
            initializer=tf.random_normal_initializer(),
            trainable=False,
            name="u",
        )
        super().build(input_shape)

    def call(self, x):
        # Flatten kernel to (in_dim, out_dim) for SN
        w_ = tf.reshape(self.w, [-1, self.filters])
        u = self.u
        for _ in range(self.power_iterations):
            v = tf.linalg.l2_normalize(tf.matmul(u, tf.transpose(w_)))
            u = tf.linalg.l2_normalize(tf.matmul(v, w_))
        self.u.assign(u)
        sigma = tf.matmul(tf.matmul(v, w_), tf.transpose(u))
        w_sn = w_ / sigma
        w_sn = tf.reshape(w_sn, tf.shape(self.w))

        y = tf.nn.conv1d(
            x, w_sn, stride=self.strides, padding=self.padding
        )
        if self.b is not None:
            y = y + self.b
        return y


def D_down_sn(h, ch: int):
    h = SNConv1D(ch, 5, strides=2, padding="same")(h)
    h = layers.LeakyReLU(0.2)(h)
    return h


def build_discriminator(cfg: CGANConfig) -> tf.keras.Model:
    assert cfg.num_classes > 0, "cfg.num_classes must be set"

    x_in = layers.Input(shape=(cfg.sig_len, 1), name="x")
    y_in = layers.Input(shape=(), dtype="int32", name="y")
    y_emb = layers.Embedding(cfg.num_classes, cfg.emb_dim, name="y_emb")(y_in)

    h = D_down_sn(x_in, 32)     # 256
    h = D_down_sn(h, 64)        # 128
    h = D_down_sn(h, 96)        # 64
    h = D_down_sn(h, 128)       # 32
    h = SNConv1D(192, 5, strides=2, padding="same")(h)  # 16
    h = layers.LeakyReLU(0.2)(h)

    feat = layers.GlobalAveragePooling1D(name="feat")(h)  # (B,C)

    fx = SNDense(cfg.emb_dim, use_bias=False, name="proj")(feat)  # (B,emb_dim)
    mul = layers.Multiply()([fx, y_emb])
    proj = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="inner")(mul)  # (B,)
    base = SNDense(1, use_bias=False, name="head")(feat)  # (B,1)
    base = layers.Flatten()(base)

    out = layers.Add(name="score")([proj, base])  # (B,)
    return models.Model([x_in, y_in], out, name="D")


def augment_real(x, jitter_std=0.01, scale_min=0.95, scale_max=1.05, shift=8):
    s = tf.random.uniform([tf.shape(x)[0], 1, 1], scale_min, scale_max)
    x = x * s + tf.random.normal(tf.shape(x), stddev=jitter_std)

    # random circular shift
    if shift and shift > 0:
        k = tf.random.uniform([], -shift, shift + 1, dtype=tf.int32)
        x = tf.roll(x, shift=k, axis=1)
    return x

class PBGAN_WGANGP(tf.keras.Model):
    def __init__(
        self,
        G: tf.keras.Model,
        D: tf.keras.Model,
        cfg: CGANConfig,
    ):
        super().__init__()
        self.G, self.D = G, D
        self.cfg = cfg
        self.gp_lambda = float(cfg.gp_lambda)
        self.drift = float(cfg.drift)

        # EMA generator (clone weights with same architecture)
        self.G_ema = build_generator(cfg)
        self.G_ema.set_weights(self.G.get_weights())
        self.ema_decay = float(cfg.ema_decay)

        self.g_opt: Optional[tf.keras.optimizers.Optimizer] = None
        self.d_opt: Optional[tf.keras.optimizers.Optimizer] = None

    def compile(self, g_opt, d_opt, **kw):
        super().compile(**kw)
        self.g_opt = g_opt
        self.d_opt = d_opt

    def _gradient_penalty(self, x_real, x_fake, y):
        eps = tf.random.uniform([tf.shape(x_real)[0], 1, 1], 0.0, 1.0)
        x_hat = eps * x_real + (1.0 - eps) * x_fake
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(x_hat)
            d_hat = self.D([x_hat, y], training=True)
        grads = gp_tape.gradient(d_hat, x_hat)
        grads = tf.reshape(grads, [tf.shape(grads)[0], -1])
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
        gp = tf.reduce_mean(tf.square(norm - 1.0))
        return gp

    def _update_ema(self):
        w = self.G.get_weights()
        w_ema = self.G_ema.get_weights()
        d = self.ema_decay
        self.G_ema.set_weights([d * we + (1.0 - d) * wi for we, wi in zip(w_ema, w)])

    def train_step(self, data):
        x_real, y = data
        y = tf.cast(y, tf.int32)
        bs = tf.shape(x_real)[0]

        # ------------------
        # 1) Train Critic D
        # ------------------
        z = tf.random.normal([bs, self.cfg.z_dim])
        with tf.GradientTape() as d_tape:
            x_real_aug = augment_real(x_real)
            x_fake = self.G([z, y], training=True)

            d_real = self.D([x_real_aug, y], training=True)
            d_fake = self.D([x_fake, y], training=True)

            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            d_loss = d_loss + self.drift * tf.reduce_mean(tf.square(d_real))

            gp = self._gradient_penalty(x_real_aug, x_fake, y)
            d_loss = d_loss + self.gp_lambda * gp

        d_grads = d_tape.gradient(d_loss, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grads, self.D.trainable_variables))

        # ------------------
        # 2) Train Generator G
        # ------------------
        z = tf.random.normal([bs, self.cfg.z_dim])
        with tf.GradientTape() as g_tape:
            x_fake = self.G([z, y], training=True)
            d_fake = self.D([x_fake, y], training=True)
            g_adv = -tf.reduce_mean(d_fake)

            # Optional feature loss (PBGAN++)
            g_loss = g_adv
        

        g_grads = g_tape.gradient(g_loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grads, self.G.trainable_variables))

        # EMA update
        self._update_ema()

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "gp": gp,
            "g_adv": g_adv,
        }


def build_encoder(cfg: CGANConfig) -> tf.keras.Model:
    assert cfg.num_classes > 0, "cfg.num_classes must be set"

    x_in = layers.Input(shape=(cfg.sig_len, 1), name="x")
    y_in = layers.Input(shape=(), dtype="int32", name="y")

    h = x_in
    # Simple downsampling encoder (kept minimal & stable)
    for ch in [64, 96, 128, 192]:
        h = layers.Conv1D(ch, 7, strides=2, padding="same")(h)
        h = layers.BatchNormalization()(h)
        h = layers.Activation("swish")(h)

    h = layers.GlobalAveragePooling1D()(h)

    y = layers.Embedding(cfg.num_classes, cfg.emb_dim)(y_in)
    y = layers.Dense(cfg.emb_dim, activation="swish")(y)

    h = layers.Concatenate()([h, y])
    z = layers.Dense(cfg.z_dim, activation=None)(h)
    return models.Model([x_in, y_in], z, name="E")

def train_encoder_E(
    G: tf.keras.Model,
    E: tf.keras.Model,
    dataset: tf.data.Dataset,
    *,
    epochs: int = 5,
    lr: float = 1e-4,
    use_ema: bool = True,
    G_ema: Optional[tf.keras.Model] = None,
    p: int = 1,
):
    G.trainable = False
    for l in G.layers:
        l.trainable = False

    if use_ema and G_ema is not None:
        G_ema.trainable = False
        for l in G_ema.layers:
            l.trainable = False

    E.trainable = True
    for l in E.layers:
        l.trainable = True

    opt = tf.keras.optimizers.Adam(lr)

    def loss_fn(x, x_hat):
        diff = tf.cast(x, tf.float32) - tf.cast(x_hat, tf.float32)
        if p == 1:
            return tf.reduce_mean(tf.abs(diff))
        elif p == 2:
            return tf.reduce_mean(tf.square(diff))
        else:
            raise ValueError("p must be 1 or 2")

    for ep in range(epochs):
        m = tf.keras.metrics.Mean()
        for batch in dataset:
            x, y = batch[0], batch[1]
            x = tf.cast(x, tf.float32)
            y = tf.cast(y, tf.int32)

            with tf.GradientTape() as tape:
                z = E([x, y], training=True)
                G_use = (G_ema if (use_ema and G_ema is not None) else G)
                x_hat = G_use([z, y], training=False)
                loss = loss_fn(x, x_hat)

            grads = tape.gradient(loss, E.trainable_variables)
            opt.apply_gradients(zip(grads, E.trainable_variables))
            m.update_state(loss)

        print(f"[E-train] epoch {ep+1}/{epochs} loss={float(m.result().numpy()):.6f}")


@tf.function
def reconstruct(x, y, G: tf.keras.Model, E: tf.keras.Model, use_ema: bool = False, G_ema: Optional[tf.keras.Model] = None):
    """x_hat = G(E(x,y), y)."""
    z = E([x, y], training=False)
    G_use = G_ema if (use_ema and G_ema is not None) else G
    x_hat = G_use([z, y], training=False)
    return x_hat


def reconstruction_error(x, x_hat, p: int = 1, reduce_axis: Tuple[int, ...] = (1, 2)) -> tf.Tensor:
    """RE(x,y) aggregated over time/channel -> (B,)."""
    diff = tf.cast(x, tf.float32) - tf.cast(x_hat, tf.float32)
    if p == 1:
        return tf.reduce_mean(tf.abs(diff), axis=reduce_axis)
    if p == 2:
        return tf.sqrt(tf.reduce_mean(tf.square(diff), axis=reduce_axis) + 1e-12)
    raise ValueError("p must be 1 or 2")


def compute_re_matrix(x: np.ndarray, G: tf.keras.Model, E: tf.keras.Model, num_classes: int, p: int = 1,
                      use_ema: bool = False, G_ema: Optional[tf.keras.Model] = None) -> np.ndarray:
    """
    RE_mat[i,c] = RE(x_i, c).
    x: (N,T,1), already scaled to [-1,1]
    """
    x_tf = tf.convert_to_tensor(x, tf.float32)
    N = int(x_tf.shape[0])
    K = int(num_classes)
    out = np.zeros((N, K), dtype=np.float32)
    for c in range(K):
        y_c = tf.fill([N], tf.cast(c, tf.int32))
        x_hat = reconstruct(x_tf, y_c, G=G, E=E, use_ema=use_ema, G_ema=G_ema)
        out[:, c] = reconstruction_error(x_tf, x_hat, p=p).numpy()
    return out


def predict_closed_set(re_mat: np.ndarray) -> np.ndarray:
    return np.argmin(re_mat, axis=1).astype(np.int64)


def accept_open_set(re_claim: np.ndarray, tau: float) -> np.ndarray:
    return (re_claim <= float(tau)).astype(np.int64)
