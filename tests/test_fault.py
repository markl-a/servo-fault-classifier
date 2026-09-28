import numpy as np
import torch

from src.features import spectral_features
from src.model import Conv1DNet
from src.simulate import CLASSES, FS, N, dataset, sample


def test_signal_shapes_and_determinism():
    a = sample("imbalance", 3, np.random.default_rng(0))
    b = sample("imbalance", 3, np.random.default_rng(0))
    assert a.shape == (N,) and np.allclose(a, b)


def test_fault_signatures_are_physical():
    rng = np.random.default_rng(1)
    def band_energy(x, lo, hi):
        s = np.abs(np.fft.rfft(x)); f = np.fft.rfftfreq(len(x), 1 / FS)
        return s[(f >= lo) & (f < hi)].sum()
    from src.simulate import machine_profile
    res = machine_profile(0)["res"]
    normal = np.mean([band_energy(sample("normal", 0, rng), res - 40, res + 40) for _ in range(10)])
    bearing = np.mean([band_energy(sample("bearing_wear", 0, rng), res - 40, res + 40) for _ in range(10)])
    assert bearing > 1.5 * normal   # bearing defects excite the machine's structural resonance


def test_split_has_no_machine_overlap():
    _, _, mtr = dataset(list(range(8)), 2, 1)
    _, _, mte = dataset([10, 11, 12], 2, 3)
    assert not set(mtr) & set(mte)


def test_model_forward_and_features():
    m = Conv1DNet(len(CLASSES))
    assert m(torch.randn(4, 1, N)).shape == (4, len(CLASSES))
    assert spectral_features(sample("normal", 0, np.random.default_rng(0))).shape == (9,)


def test_onnx_export_matches_torch_if_trained():
    """Runs after `python -m src.train`; checks the deployable artifact gives the same logits as PyTorch."""
    import os
    import pytest
    if not os.path.exists("results/cnn_envelope.onnx"):
        pytest.skip("train first")
    import onnxruntime as ort
    m = Conv1DNet(len(CLASSES), in_ch=2)
    m.load_state_dict(torch.load("results/cnn_envelope.pt")); m.eval()
    x = torch.randn(3, 2, N)
    ref = m(x).detach().numpy()
    out = ort.InferenceSession("results/cnn_envelope.onnx").run(None, {"signal": x.numpy()})[0]
    assert np.allclose(ref, out, atol=1e-4)
