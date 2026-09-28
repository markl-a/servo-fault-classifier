"""Hand-crafted baseline: log magnitude spectrum in orders of shaft speed is unknown, so use fixed bands + envelope spectrum stats."""
import numpy as np
from scipy.signal import hilbert

from .simulate import FS


def spectral_features(x: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    bands = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1000)]
    band_e = [np.log1p(spec[(freqs >= a) & (freqs < b)].sum()) for a, b in bands]
    env = np.abs(hilbert(x))
    env_spec = np.abs(np.fft.rfft(env - env.mean()))
    return np.array(band_e + [x.std(), np.abs(x).max() / (x.std() + 1e-9), np.log1p(env_spec[:300].max())], dtype=np.float32)
