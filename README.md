# servo-fault-classifier

**Fault classification for servo/linear axes from vibration signals — and an honest comparison: hand-crafted spectral features vs a 1D CNN, evaluated on machines the models have never seen.**

Portfolio demo (Sept 2026). Signals are **synthetic but physically motivated** (shaft harmonics, bearing-defect modulation of a structural resonance, looseness harmonics). Trains on CPU in under a minute. Exports ONNX for edge deployment.

## Result (test = 3 unseen machines, 600 samples)

| Model | Macro-F1 | bearing_wear F1 | normal F1 | Params | CPU latency (batch 1) |
|---|---|---|---|---|---|
| Spectral bands + envelope stats → logistic regression | **0.987** | — | — | 9 features | — |
| 1D CNN on raw signal | 0.872 | 0.681 | 0.739 | 59 k | ~2 ms |
| 1D CNN on raw + Hilbert envelope (2 channels) | 0.954 | 0.884 | 0.909 | 60 k | ~1–2 ms |

Numbers are from `results/report.json` of the committed run.

### What this shows

- **The split matters.** Train on machines 0–7, select on 8–9, test on 10–12. Each machine has its own gain, noise and resonance frequency; a random split would leak those into the test set and overstate accuracy.
- **The raw CNN overfit to machine-specific resonance.** It confused *bearing wear* with *normal* on new machines (58 of 120 bearing samples → normal). Bearing defects show up as *modulation* of whatever resonance a machine has; the raw CNN learned *where* the resonance was on the training machines.
- **Giving the model the right physics fixes most of it.** Adding the demodulated envelope as a second channel lifts macro-F1 from 0.872 to 0.954 and bearing-wear F1 from 0.68 to 0.88.
- **The simple baseline still wins here** — its features are machine-invariant by construction. In production I would start from the feature model, use the CNN where features plateau, and keep both behind the same evaluation split.

## Run

```bash
pip install -r requirements.txt
python -m src.train          # trains LR baseline + two CNNs, writes results/report.json and cnn_envelope.onnx
pytest -q                    # signal physics, no machine leakage, model shapes, ONNX == PyTorch logits
```

## Deploying to an edge controller

`results/cnn_envelope.onnx` runs with ONNX Runtime on CPU; the next step on an ARM/NPU target would be INT8 quantisation with a calibration set and a per-class accuracy regression check against this report before shipping.

## Limits

Synthetic data. Real deployments need labelled field data per axis type, sensor placement control, and drift monitoring — the machine-level split here is the minimum bar for that evaluation.
