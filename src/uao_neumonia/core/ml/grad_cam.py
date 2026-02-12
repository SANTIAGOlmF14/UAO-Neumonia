from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from uao_neumonia.core.ml.preprocess_img import preprocess


def grad_cam(array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str = "conv10_thisone") -> np.ndarray:
    img = preprocess(array)
    preds = model.predict(img)
    argmax = int(np.argmax(preds[0]))
    output = model.output[:, argmax]
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img)

    for filters in range(64):
        conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = (heatmap * hif).astype(np.uint8)

    superimposed_img = cv2.add(transparency, img2).astype(np.uint8)
    return superimposed_img[:, :, ::-1]
