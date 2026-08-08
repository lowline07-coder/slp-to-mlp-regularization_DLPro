# Breast Cancer Classification — SLP → MLP → Regularization

A TensorFlow/Keras walkthrough on the Breast Cancer Wisconsin (Diagnostic) dataset, building up from a single-neuron baseline to a fully regularized network.

## What's inside

`DL_PR1.ipynb` runs top to bottom, no manual edits needed:

1. **Setup** — installs TensorFlow (fixes `ModuleNotFoundError: No module named 'tensorflow'`) and imports everything else.
2. **Data Loading, EDA & Preprocessing** — loads `wdbc.data`, checks class balance and feature correlation, does a stratified 80/20 split, and scales features with `StandardScaler`.
3. **Single-Layer Perceptron (SLP)** — a one-neuron linear baseline.
4. **Multi-Layer Perceptron (MLP)** — a 64→32→1 network, comparing ReLU / Tanh / Sigmoid hidden activations.
5. **Early Stopping** — trains a 128→64→1 MLP long enough to overfit, then shows how `EarlyStopping` recovers the best generalizing weights.
6. **Dropout** — adds Dropout to the same architecture and compares rates 0.1 / 0.3 / 0.5.
7. **Regularization** — compares L1, L2, and L1-L2 (ElasticNet) weight penalties.
8. **Final Model & Results** — combines Dropout + L2 + Early Stopping into one model, builds a full comparison table, and gives a clinical deployment recommendation.

## Running it

```bash
jupyter notebook DL_PR1.ipynb
```

If `tensorflow` isn't installed, the first code cell (`%pip install -q tensorflow`) handles it automatically — just run all cells in order (**Kernel → Restart & Run All**).

Alternatively, install everything up front:

```bash
pip install -r requirements.txt
```

## Files

| File | Description |
|---|---|
| `DL_PR1.ipynb` | The notebook |
| `wdbc.data` / `wdbc.names` | Raw UCI dataset and column descriptions |
| `requirements.txt` | Pinned dependencies |
| `results_comparison_table.csv` | Test-set metrics for every model (regenerated on each run) |
| `plots/` | Saved figures (regenerated on each run) |

## Results summary

| Model | Regularization | Dropout | Early Stopping |
|---|---|---|---|
| SLP | — | — | No |
| MLP (best activation) | — | — | No |
| MLP + Early Stopping | — | — | Yes |
| MLP + Dropout (best rate) | — | ✓ | Yes |
| MLP + L2 | L2 | — | Yes |
| **Final Combined Model** | L2 | ✓ | Yes |

The Final Combined Model (Dropout + L2 + Early Stopping) gives the smallest train/validation gap and is the recommended candidate for deployment — see Section 7 of the notebook for the full metrics and threshold discussion. Exact numbers vary slightly between runs since weights are randomly initialized.

## Key takeaways

- **Scaling** matters for gradient-based optimization — unscaled features (e.g. `area_mean` vs `fractal_dimension_mean`) distort and slow convergence.
- An **SLP** is linear and hits a hard accuracy ceiling; hidden layers (**MLP**) let the model learn non-linear decision boundaries.
- **ReLU** converges fastest among the tested hidden activations for this tabular task.
- **Early Stopping** prevents wasted, harmful over-training by restoring the best-validation-loss weights.
- **Dropout** (rate ≈0.3) and **L2** regularization each reduce overfitting; combining them gives the most reliable generalization.
- In a clinical setting, minimizing **false negatives** (missed malignancies) matters more than raw accuracy — the classification threshold should be tuned accordingly, and any model should support rather than replace clinician judgment.

## Dataset

Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository — 569 samples, 30 numeric features, binary target (0 = Malignant, 1 = Benign).
