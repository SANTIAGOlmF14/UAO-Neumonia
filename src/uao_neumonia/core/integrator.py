from __future__ import annotations

import numpy as np

from uao_neumonia.core.ml.grad_cam import grad_cam
from uao_neumonia.core.ml.load_model import get_model
from uao_neumonia.core.ml.preprocess_img import preprocess


def predict(array):
    batch_array_img = preprocess(array)
    model = get_model()

    preds = model.predict(batch_array_img)
    prediction = int(np.argmax(preds))
    proba = float(np.max(preds) * 100)

    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"

    heatmap = grad_cam(array, model=model)
    return label, proba, heatmap
