import numpy as np
import pytest

import uao_neumonia.core.integrator as integrator


class _FakeModel:
    def __init__(self, preds: np.ndarray):
        self._preds = preds

    def predict(self, _batch):
        return self._preds


@pytest.mark.parametrize(
    ("preds", "expected_label", "expected_proba"),
    [
        (np.array([[0.9, 0.05, 0.05]], dtype=float), "bacteriana", 90.0),
        (np.array([[0.1, 0.8, 0.1]], dtype=float), "normal", 80.0),
        (np.array([[0.2, 0.3, 0.5]], dtype=float), "viral", 50.0),
    ],
)
def test_predict_maps_label_and_probability(monkeypatch, preds, expected_label, expected_proba):
    # Evita depender de OpenCV/Tensorflow real: mockeamos todo lo pesado.
    monkeypatch.setattr(integrator, "preprocess", lambda _array: np.zeros((1, 512, 512, 1), dtype=float))
    monkeypatch.setattr(integrator, "get_model", lambda: _FakeModel(preds))

    expected_heatmap = np.zeros((512, 512, 3), dtype=np.uint8)
    monkeypatch.setattr(integrator, "grad_cam", lambda _array, model: expected_heatmap)

    label, proba, heatmap = integrator.predict(np.zeros((10, 10, 3), dtype=np.uint8))

    assert label == expected_label
    assert proba == pytest.approx(expected_proba, abs=1e-6)
    assert heatmap is expected_heatmap
