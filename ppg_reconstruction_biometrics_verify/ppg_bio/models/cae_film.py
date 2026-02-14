from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# -------------------------
# Config
# -------------------------

@dataclass
class CAEConfig:
    sig_len: int = 512
    num_classes: int = 0          # MUST be set
    emb_dim: int = 64
    latent_dim: int = 32
    dropout: float = 0.1


# -------------------------
# Conditioning: FiLM
# -------------------------

class FiLM(layers.Layer):
    """Feature-wise Linear Modulation: h' = gamma(y) * h + beta(y)."""
    def __init__(self, ch: int, **kwargs):
        super().__init__(**kwargs)
        self.ch = int(ch)
        self.gamma = layers.Dense(self.ch, name="gamma")
        self.beta = layers.Dense(self.ch, name="beta")

    def call(self, h: tf.Tensor, y_emb: tf.Tensor) -> tf.Tensor:
        # h: (B,T,C), y_emb: (B,E)
        g = self.gamma(y_emb)  # (B,C)
        b = self.beta(y_emb)   # (B,C)
        T = tf.shape(h)[1]
        g = tf.tile(g[:, None, :], [1, T, 1])
        b = tf.tile(b[:, None, :], [1, T, 1])
        return g * h + b

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ch": self.ch})
        return cfg


def _embed_y(y_in: tf.Tensor, num_classes: int, emb_dim: int, name: str) -> tf.Tensor:
    y = layers.Embedding(num_classes, emb_dim, name=f"{name}_emb")(y_in)
    y = layers.Dense(emb_dim, activation="swish", name=f"{name}_proj")(y)
    return y  # (B, emb_dim)


# -------------------------
# Encoder / Decoder blocks
# -------------------------

def enc_block(x: tf.Tensor, ch: int, dropout: float, name: str):
    """Conv1D-BN-Swish -> Conv1D-BN-Swish -> MaxPool, returns (x_pool, x_skip)."""
    h = layers.Conv1D(ch, 5, padding="same", name=f"{name}_conv1")(x)
    h = layers.BatchNormalization(name=f"{name}_bn1")(h)
    h = layers.Activation("swish", name=f"{name}_act1")(h)

    h = layers.Conv1D(ch, 5, padding="same", name=f"{name}_conv2")(h)
    h = layers.BatchNormalization(name=f"{name}_bn2")(h)
    h = layers.Activation("swish", name=f"{name}_act2")(h)

    if dropout and dropout > 0:
        h = layers.Dropout(dropout, name=f"{name}_drop")(h)

    x_pool = layers.MaxPooling1D(pool_size=2, name=f"{name}_pool")(h)
    return x_pool, h


def dec_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    y_emb: tf.Tensor,
    ch: int,
    dropout: float,
    name: str,
):
    """
    UpSample -> concat skip -> Conv1D-BN-Swish -> FiLM -> Conv1D-BN-Swish
    """
    h = layers.UpSampling1D(size=2, name=f"{name}_up")(x)
    h = layers.Concatenate(name=f"{name}_concat")([h, skip])

    h = layers.Conv1D(ch, 5, padding="same", name=f"{name}_conv1")(h)
    h = layers.BatchNormalization(name=f"{name}_bn1")(h)
    h = layers.Activation("swish", name=f"{name}_act1")(h)

    # Condition via FiLM (paper-consistent)
    h = FiLM(ch, name=f"{name}_film")(h, y_emb)

    h = layers.Conv1D(ch, 5, padding="same", name=f"{name}_conv2")(h)
    h = layers.BatchNormalization(name=f"{name}_bn2")(h)
    h = layers.Activation("swish", name=f"{name}_act2")(h)

    if dropout and dropout > 0:
        h = layers.Dropout(dropout, name=f"{name}_drop")(h)

    return h


# -------------------------
# Model builder
# -------------------------

def build_cae_unet_film(cfg: CAEConfig, with_aux_head: bool = True) -> tf.keras.Model:
    """
    Build conditional AE U-Net with FiLM conditioning.

    Inputs:
      - x: (B, T, 1)  PPG signal
      - y: (B,) int32 identity label (condition)

    Outputs:
      - x_hat: (B, T, 1) reconstruction
      - (optional) id_logits: (B, K) auxiliary head from x-only bottleneck
    """
    assert cfg.num_classes > 0, "cfg.num_classes must be set"

    x_in = layers.Input(shape=(cfg.sig_len, 1), name="x")
    y_in = layers.Input(shape=(), dtype="int32", name="y")

    y_emb = _embed_y(y_in, cfg.num_classes, cfg.emb_dim, name="y")

    # Encoder: 512→256→128→64→32
    x1, s1 = enc_block(x_in, 32,  cfg.dropout, name="enc1")
    x2, s2 = enc_block(x1,   64,  cfg.dropout, name="enc2")
    x3, s3 = enc_block(x2,   96,  cfg.dropout, name="enc3")
    x4, s4 = enc_block(x3,   128, cfg.dropout, name="enc4")

    # Bottleneck (conditional stream for decoder)
    z = layers.Conv1D(cfg.latent_dim, 3, padding="same", name="bottleneck")(x4)
    z = layers.BatchNormalization(name="latent_bn")(z)
    if cfg.dropout and cfg.dropout > 0:
        z = layers.Dropout(cfg.dropout, name="latent_drop")(z)

    # Decoder mirrors encoder with FiLM
    d1 = dec_block(z,  s4, y_emb, 128, cfg.dropout, name="dec1")  # 32→64
    d2 = dec_block(d1, s3, y_emb, 96,  cfg.dropout, name="dec2")  # 64→128
    d3 = dec_block(d2, s2, y_emb, 64,  cfg.dropout, name="dec3")  # 128→256
    d4 = dec_block(d3, s1, y_emb, 32,  cfg.dropout, name="dec4")  # 256→512

    x_hat = layers.Conv1D(1, 7, padding="same", activation=None, name="recon")(d4)

    if not with_aux_head:
        return models.Model([x_in, y_in], x_hat, name="CAE_FiLM")

    # Auxiliary ID head from x-only bottleneck (regularizer during training)
    z_x = layers.Conv1D(cfg.latent_dim, 1, padding="same", name="bottleneck_x")(x4)
    z_x = layers.BatchNormalization(name="latent_x_bn")(z_x)
    z_x = layers.Activation("swish", name="latent_x_act")(z_x)
    h = layers.GlobalAveragePooling1D(name="aux_gap")(z_x)
    h = layers.Dense(96, activation="swish", name="aux_fc")(h)
    id_logits = layers.Dense(cfg.num_classes, activation=None, name="aux_logits")(h)

    return models.Model([x_in, y_in], [x_hat, id_logits], name="CAE_FiLM_Aux")


# -------------------------
# Inference helpers
# -------------------------

def reconstruct(
    x: np.ndarray | tf.Tensor,
    y: np.ndarray | tf.Tensor,
    model: tf.keras.Model,
) -> tf.Tensor:
    """
    Run CAE reconstruction.

    x: (B,T,1)
    y: (B,)
    returns x_hat: (B,T,1)
    """
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y, dtype=tf.int32)

    out = model([x_tf, y_tf], training=False)
    x_hat = out[0] if isinstance(out, (list, tuple)) else out
    return x_hat


def reconstruction_error(
    x: np.ndarray | tf.Tensor,
    x_hat: np.ndarray | tf.Tensor,
    p: int = 1,
    reduce_axis: Tuple[int, ...] = (1, 2),
) -> tf.Tensor:
    """
    RE(x,y) = ||x - x_hat||_p aggregated over time/channel.

    Returns: (B,)
    """
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    xh_tf = tf.convert_to_tensor(x_hat, dtype=tf.float32)
    diff = x_tf - xh_tf

    if p == 1:
        return tf.reduce_mean(tf.abs(diff), axis=reduce_axis)
    if p == 2:
        return tf.sqrt(tf.reduce_mean(tf.square(diff), axis=reduce_axis) + 1e-12)
    raise ValueError("p must be 1 or 2")


def compute_re_matrix(
    x: np.ndarray,
    model: tf.keras.Model,
    num_classes: int,
    p: int = 1,
) -> np.ndarray:
    """
    Compute RE for all claimed identities:
        RE_mat[i, c] = RE(x_i, c)

    x: (N,T,1)
    returns: (N, K)
    """
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    N = int(x_tf.shape[0])
    K = int(num_classes)
    re_mat = np.zeros((N, K), dtype=np.float32)

    for c in range(K):
        y_c = tf.fill([N], tf.cast(c, tf.int32))
        x_hat = reconstruct(x_tf, y_c, model)
        re = reconstruction_error(x_tf, x_hat, p=p)
        re_mat[:, c] = re.numpy()

    return re_mat


def predict_closed_set(re_mat: np.ndarray) -> np.ndarray:
    """Closed-set identification: choose identity with minimum RE."""
    return np.argmin(re_mat, axis=1).astype(np.int64)


def accept_open_set(re_claim: np.ndarray, tau: float) -> np.ndarray:
    """Open-set verification: accept if RE <= tau."""
    return (re_claim <= float(tau)).astype(np.int64)
