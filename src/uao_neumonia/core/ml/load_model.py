from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from uao_neumonia.utils.paths import repo_root


_MODEL = None


def get_model(model_path: str | Path | None = None):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if model_path is None:
        model_path = repo_root() / "model" / "conv_MLP_84.h5"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    _MODEL = tf.keras.models.load_model(str(model_path), compile=False)
    return _MODEL
