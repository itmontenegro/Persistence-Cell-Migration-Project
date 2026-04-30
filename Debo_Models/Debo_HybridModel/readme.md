````markdown
# Universal Migration Prediction (Hybrid Model)

This repository contains the code for training and evaluating a physics-guided Hybrid PyTorch model to predict migration parameters (`Dr` and `H`) from trajectory data. The codebase is fully optimized to automatically detect and use a CUDA-enabled GPU on Linux if one is available.

---

## 1. Project Directory Structure

Before starting, ensure your project folder is structured exactly like this:

```text
project_root/
│
├── data/
│   └── Data_Alpha_0_25/         # Place your CSV data folders inside here
│
├── src/
│   ├── dataset.py             # Dataset loader
│   └── models.py              # Hybrid model architecture
│
├── train.py                   # Training script
├── evaluate_and_visualize.py  # Inference and plotting script
└── requirements.txt             # Python dependencies
````

## 2. Environment Setup (Linux)

Open your terminal, navigate to your `project_root` directory, and follow these steps:

### Step A: Create the virtual environment

We will create a virtual environment named `hybrid`:

```bash
python3 -m venv hybrid
```

### Step B: Activate the virtual environment

```bash
source hybrid/bin/activate
```

> You will know it worked if your terminal prompt starts with `(hybrid)`.

### Step C: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Training the Model

Run the training script:

```bash
python train.py --data_path "data/Data_Alpha_0_25" --epochs 100 --batch_size 16 --lr 0.0001
```

## 4. Evaluation & Visualization

After training, evaluate the model using:

```bash
python evaluate_and_visualize.py --data_path "data/Data_Alpha_0_25" --checkpoint "saved_models/best_hybrid_Dr_H_2.pth" --batch_size 16
```

### Output files:

* `metrics_phase2_hybrid_Dr_H.txt`
  → Contains RMSE, MAE, and R² scores

* `ladder_phase2_hybrid_Dr.png`
  → Predicted vs actual plot for **Dr**

* `ladder_phase2_hybrid_H.png`
  → Predicted vs actual plot for **H**

---

## 5. Shutting Down

To exit the virtual environment:

```bash
deactivate
```