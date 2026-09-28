"""Train and compare: (1) spectral features + logistic regression, (2) 1D CNN on raw signal.

Split BY MACHINE: train on machines 0-7, validate on 8-9, test on unseen machines 10-12.
A random split would leak each machine's gain/noise/resonance into the test set and overstate accuracy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.signal import hilbert
from torch import nn

from .features import spectral_features
from .model import Conv1DNet
from .simulate import CLASSES, dataset

ROOT = Path(__file__).resolve().parent.parent


def main(epochs: int = 12, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    Xtr, ytr, _ = dataset(list(range(8)), per_class=40, seed=1)
    Xva, yva, _ = dataset([8, 9], per_class=30, seed=2)
    Xte, yte, _ = dataset([10, 11, 12], per_class=40, seed=3)

    # baseline
    ftr, fte = np.stack([spectral_features(x) for x in Xtr]), np.stack([spectral_features(x) for x in Xte])
    sc = StandardScaler().fit(ftr)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(ftr), ytr)
    base_pred = lr.predict(sc.transform(fte))

    def z(X):
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)

    def to_tensor(X, envelope: bool):
        chans = [z(X)]
        if envelope:  # demodulated envelope: bearing defects show up as modulation, independent of where the resonance sits
            chans.append(z(np.abs(hilbert(X, axis=1))))
        return torch.from_numpy(np.stack(chans, 1).astype(np.float32))

    def fit_cnn(envelope: bool):
        torch.manual_seed(seed)
        Ttr, Tva, Tte = (to_tensor(X, envelope) for X in (Xtr, Xva, Xte))
        model = Conv1DNet(len(CLASSES), in_ch=Ttr.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        lossf, ytr_t = nn.CrossEntropyLoss(), torch.tensor(ytr)
        best, best_state, hist = -1, None, []
        t0 = time.perf_counter()
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(Ttr))
            for i in range(0, len(perm), 32):
                idx = perm[i:i + 32]
                loss = lossf(model(Ttr[idx]), ytr_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                va_f1 = f1_score(yva, model(Tva).argmax(1).numpy(), average="macro")
            hist.append({"epoch": ep, "loss": round(loss.item(), 4), "val_macro_f1": round(va_f1, 4)})
            if va_f1 > best:  # model selection on validation machines, never on test
                best, best_state = va_f1, {k: v.clone() for k, v in model.state_dict().items()}
        train_s = time.perf_counter() - t0
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            pred = model(Tte).argmax(1).numpy()
            t1 = time.perf_counter(); model(Tte[:1]); lat_ms = (time.perf_counter() - t1) * 1000
        summary = {"macro_f1": round(f1_score(yte, pred, average="macro"), 4), "params": sum(p.numel() for p in model.parameters()),
                   "train_seconds": round(train_s, 1), "cpu_latency_ms_batch1": round(lat_ms, 2), "best_val_macro_f1": round(best, 4),
                   "per_class_f1": {c: round(v["f1-score"], 3) for c, v in classification_report(yte, pred, target_names=CLASSES, output_dict=True, zero_division=0).items() if c in CLASSES},
                   "confusion": confusion_matrix(yte, pred).tolist()}
        return model, Tte, summary, hist

    raw_model, _, raw_sum, raw_hist = fit_cnn(envelope=False)
    model, Tte, env_sum, env_hist = fit_cnn(envelope=True)
    torch.save(model.state_dict(), ROOT / "results" / "cnn_envelope.pt")
    torch.onnx.export(model, Tte[:1], ROOT / "results" / "cnn_envelope.onnx", input_names=["signal"], output_names=["logits"],
                      dynamic_axes={"signal": {0: "batch"}}, opset_version=17, dynamo=False)
    report = {
        "split": "by machine: train 0-7, val 8-9, test 10-12 (unseen)",
        "labels": CLASSES,
        "baseline_logreg_spectral": {"macro_f1": round(f1_score(yte, base_pred, average="macro"), 4),
                                     "confusion": confusion_matrix(yte, base_pred).tolist()},
        "cnn_raw": raw_sum,
        "cnn_raw_plus_envelope": env_sum,
        "history": {"cnn_raw": raw_hist, "cnn_raw_plus_envelope": env_hist},
    }
    (ROOT / "results" / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    r = main()
    for k in ("baseline_logreg_spectral", "cnn_raw", "cnn_raw_plus_envelope"):
        print(k, {kk: vv for kk, vv in r[k].items() if kk != "confusion"})
