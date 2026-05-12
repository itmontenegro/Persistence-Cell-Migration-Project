# TCN-Implementation 

This repository contains the code for training and evaluating a **Temporal Convolutional Network (TCN)** to jointly predict two biophysical migration parameters(currently) — rotational diffusion coefficient (`Dr`) and persistence parameter (`H`) — directly from simulated 2D cell trajectory data.



---

## 1. Project Directory Structure

```
project_root/
│
├── data/                        # Simulation trajectory CSVs
│   └── Alpha_{a}__Dr_{d}__H_{h}/
│       └── Sim_*.csv
│
├── TCN_Base.py                  # Model, dataset, training, evaluation
├── base_tcn_7in.pth             # Best checkpoint (generated after training)
├── requirements.txt
└── visualize.py                # Plots generation
```

---

## 2. Environment Setup

**Step A: Create a virtual environment**
```
python3 -m venv tcn_env
```

**Step B: Activate it**
```
source tcn_env/bin/activate
```
Your prompt will change to `(tcn_env)`.

**Step C: Install dependencies**
```
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

---

## 3. Dataset

Simulation CSV files follow the directory naming convention `Alpha_{a}__Dr_{d}__H_{h}/Sim_*.csv`, where underscores replace decimal points. Each CSV contains columns `X`, `Y`, `Theta` for position and heading angle at each timestep (up to 1001 steps per trajectory). `Dr` and `H` values are parsed directly from the directory path via regex.

---

## 4. Feature Engineering

Each trajectory is transformed into a **7-dimensional feature sequence** `[T × 7]` before being fed to the model:

| Feature | Description |
|---|---|
| `dx`, `dy` | Per-step displacements |
| `step_len` | Euclidean step length `√(dx²+dy²)` |
| `efficiency` | `net_disp / (cum_dist + ε)` — path straightness |
| `sin(θ)`, `cos(θ)` | Circular encoding of heading angle (avoids 0/2π discontinuity) |
| `net_disp_norm` | Net displacement normalized by per-trajectory maximum |

`dx`, `dy`, `step_len`, and `efficiency` are standardized using `StandardScaler` fitted on training data only. `Dr` is log-transformed (`log10(Dr+1)`) before target scaling to handle its wide dynamic range; the inverse transform is applied at evaluation.

---

## 5. Architecture

**`BaseTCNRegressor`** — 6 stacked `CausalResidualBlock` layers followed by global average pooling and a linear output head.

### CausalResidualBlock

Each block applies a **dilated causal 1D convolution**, trims future-context padding, applies GroupNorm (equivalent to LayerNorm over channels), ReLU, and Dropout, then adds a residual connection. A 1×1 pointwise convolution is applied on the residual path when channel dimensions change.

Causal padding of `(kernel_size − 1) × dilation` is added before each convolution and the corresponding tail is trimmed afterward, ensuring the output at time *t* never sees future inputs.

### Block Configuration

| Block | In → Out Channels | Dilation |
|---|---|---|
| 1 | 7 → 64 | 1 |
| 2 | 64 → 128 | 2 |
| 3 | 128 → 128 | 4 |
| 4 | 128 → 256 | 8 |
| 5 | 256 → 256 | 16 |
| 6 | 256 → 512 | 32 |

**Receptive field:** RF = 1 + 2 × (1+2+4+8+16+32) = **127 timesteps**

After the 6 residual blocks, global average pooling over the time dimension collapses [B × 512 × T] → [B × 512], making the representation independent of variable sequence lengths introduced by padding. A final Linear(512, 2) outputs normalized predictions for (Dr, H).

**Total parameters:** ~937,000

---

## 6. Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Max epochs | 200 |
| Early stopping patience | 25 epochs |
| Scheduler | ReduceLROnPlateau (patience=10) |
| Loss function | HuberLoss |
| Dropout | 0.2 |
| Kernel size | 3 |
| Input features | 7 |
| Output targets | 2 (Dr, H) |
| Max sequence length | 1001 steps |
| Random seed | 42 |


Why Huber loss?- Targets are standardized (zero-mean, unit-variance via `StandardScaler`), so most residuals fall well within the quadratic region (|error| ≤ 1.0, δ = 1.0 default), giving MSE-like smooth gradient flow near convergence. The linear regime (|error| > 1.0) acts as a safety net against the few large residuals from boundary-regime trajectories early in training, preventing them from dominating gradients. Essentially MSE for this case — until an outlier decides otherwise.
---

## 7. Data Split Strategy

Stratified by `(Dr, H)` combination string to ensure every biophysical regime is proportionally represented across all three splits:

| Split | Proportion |
|---|---|
| Train | ~72.25% |
| Validation | ~12.75% |
| Test | 15% |

Scalers are fitted exclusively on training data (`partial_fit` per trajectory for inputs; `fit` on aggregated log-targets) to prevent data leakage.

---

## 8. Training

Place simulation data under the project root directory and run:

```
python TCN_Base.py
```

The script discovers all `Sim_*.csv` files recursively (modify `base_path` in `DataManager` if your data lives elsewhere). Training logs every 5 epochs. The best checkpoint by validation loss is saved to `base_tcn_7in.pth`. Training halts early if validation loss does not improve for 25 consecutive epochs.

Device: Apple Silicon MPS by default. For NVIDIA GPUs, change the device line in `run_pipeline()` to use `torch.cuda`.

---

## 9. Results

Evaluated on the held-out test set in original physical units after inverse-transforming all predictions.

| Target | R² | MAE | MSE |
|---|---|---|---|
| Dr (Rotational Diffusion) | **0.99903** | 0.490372 | 1.136213 |
| H (Persistence Parameter) | **0.99863** | 0.009577 | 0.000166 |

Both targets exceed R² > 0.998. Minor scatter on `Dr` at the highest-Dr regime is expected — high rotational diffusion degrades trajectory coherence(Something Nacho said), making estimation intrinsically noisier at that boundary. 

---

## 10. Reproducibility

Full reproducibility is enforced via `seed_everything(42)`, which seeds Python's `random`, `PYTHONHASHSEED`, NumPy, PyTorch (CPU + CUDA), and sets `cudnn.deterministic = True` / `cudnn.benchmark = False`. Both `train_test_split` calls use `random_state=42`. Pin exact package versions with `pip freeze` for bit-exact reproduction across machines.

---

## 11. Shutting Down

```
deactivate
```
