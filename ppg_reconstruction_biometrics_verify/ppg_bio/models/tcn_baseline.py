from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def residual_block(x: tf.Tensor, filters: int, k: int = 3, d: int = 1) -> tf.Tensor:
    s = x
    x = layers.Conv1D(filters, k, padding="causal", dilation_rate=d, activation="relu")(x)
    x = layers.Conv1D(filters, k, padding="causal", dilation_rate=d)(x)
    if s.shape[-1] != x.shape[-1]:
        s = layers.Conv1D(filters, 1, padding="same")(s)
    x = layers.Add()([x, s])
    x = layers.Activation("relu")(x)
    return x


def build_embedding_classifier(
    sig_len: int,
    num_classes: int,
    emb_dim: int,
    filters: int = 64,
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16),
) -> tf.keras.Model:
    inp = layers.Input(shape=(sig_len, 1), name="ppg")
    x = inp
    for d in dilations:
        x = residual_block(x, filters, k=3, d=d)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc")(x)

    emb = layers.Dense(emb_dim, activation=None, name="embedding")(x)
    emb = layers.UnitNormalization(axis=-1, name="emb_l2")(emb)

    probs = layers.Dense(num_classes, activation="softmax", name="softmax")(emb)
    return models.Model(inp, [emb, probs], name="tcn_embed_cls")


def extract_embeddings(model: tf.keras.Model, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    embs = []
    for i in range(0, len(X), batch):
        e, _ = model.predict(X[i:i + batch], verbose=0)
        embs.append(e)
    embs = np.vstack(embs).astype(np.float32)
    # safety normalize
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    return embs


def predict_closed_set_softmax(model: tf.keras.Model, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    preds = []
    for i in range(0, len(X), batch):
        _, p = model.predict(X[i:i + batch], verbose=0)
        preds.append(np.argmax(p, axis=1))
    return np.concatenate(preds, axis=0).astype(np.int64)

def build_prototypes(emb_tr: np.ndarray, y_tr: np.ndarray, num_classes: int) -> np.ndarray:
    protos = np.zeros((num_classes, emb_tr.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        m = emb_tr[y_tr == c]
        if len(m) == 0:
            raise ValueError(f"No samples for class {c} in training.")
        v = np.mean(m, axis=0)
        protos[c] = v / (np.linalg.norm(v) + 1e-9)
    return protos


def predict_closed_set_prototype(emb: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    # emb/prototypes are already L2-normalized => dot = cosine
    scores = emb @ prototypes.T
    return np.argmax(scores, axis=1).astype(np.int64)
