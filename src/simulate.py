"""Synthetic servo-axis vibration signals with physically motivated fault signatures.

Each sample is 1 s of acceleration at 2 kHz on a linear axis driven at shaft speed f0.
- normal:        1x shaft harmonic + broadband noise
- imbalance:     strong 1x
- misalignment:  strong 2x (and some 3x)
- bearing_wear:  high-frequency bursts amplitude-modulated at a bearing defect frequency
- looseness:     many harmonics + clipping-like impacts
Machines differ in f0, gain and noise so we can split by machine and test generalisation.
"""
from __future__ import annotations

import numpy as np

CLASSES = ["normal", "imbalance", "misalignment", "bearing_wear", "looseness"]
FS, N = 2000, 2000


def machine_profile(mid: int) -> dict:
    r = np.random.default_rng(1000 + mid)
    return {"f0": r.uniform(18, 32), "gain": r.uniform(0.7, 1.3), "noise": r.uniform(0.15, 0.35), "res": r.uniform(550, 750)}


def sample(cls: str, mid: int, rng: np.random.Generator) -> np.ndarray:
    p = machine_profile(mid)
    t = np.arange(N) / FS
    f0 = p["f0"] * rng.uniform(0.97, 1.03)
    x = 0.3 * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28))
    if cls == "imbalance":
        x += rng.uniform(0.9, 1.4) * np.sin(2 * np.pi * f0 * t)
    elif cls == "misalignment":
        x += rng.uniform(0.7, 1.1) * np.sin(2 * np.pi * 2 * f0 * t) + 0.3 * np.sin(2 * np.pi * 3 * f0 * t)
    elif cls == "bearing_wear":
        bpfo = f0 * rng.uniform(3.2, 3.8)
        env = (1 + np.sin(2 * np.pi * bpfo * t)) / 2
        x += rng.uniform(0.4, 0.8) * env * np.sin(2 * np.pi * p["res"] * t)
    elif cls == "looseness":
        for k in range(1, 9):
            x += rng.uniform(0.1, 0.3) / k ** 0.5 * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28))
        x = np.clip(x, -0.6, 0.6) * 1.6
    x = p["gain"] * x + rng.normal(0, p["noise"], N)
    return x.astype(np.float32)


def dataset(machines: list[int], per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y, m = [], [], []
    for mid in machines:
        for ci, c in enumerate(CLASSES):
            for _ in range(per_class):
                X.append(sample(c, mid, rng)); y.append(ci); m.append(mid)
    return np.stack(X), np.array(y), np.array(m)
