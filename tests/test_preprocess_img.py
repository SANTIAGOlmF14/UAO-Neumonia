import numpy as np

from uao_neumonia.core.ml.preprocess_img import preprocess


def test_preprocess_returns_expected_shape_and_range():
    rng = np.random.default_rng(0)
    # Imagen RGB simulada (alto/ ancho distintos a 512 para validar resize)
    img = rng.integers(0, 256, size=(700, 600, 3), dtype=np.uint8)

    out = preprocess(img)

    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 512, 512, 1)

    # Debe estar normalizado en [0, 1]
    assert np.min(out) >= 0.0
    assert np.max(out) <= 1.0
