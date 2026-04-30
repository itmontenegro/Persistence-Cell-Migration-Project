
# Universal Migration Prediction (Hybrid Model)

This repository contains the code for training and evaluating a physics-guided Hybrid PyTorch model to predict migration parameters (`Dr` and `H`) from trajectory data. The codebase is optimized to automatically detect and use a CUDA-enabled GPU on Linux if available.

---

## 1. Project Directory Structure

Before starting, ensure your project folder is structured exactly like this:

```text
project_root/
│
├── data/
│   └── Data_Alpha_0_25/
│
├── src/
│   ├── dataset.py
│   └── models.py
│
├── train.py
├── evaluate_and_visualize.py
└── requirements.txt
```

---

## 2. Environment Setup (Linux)

Open your terminal and navigate to the `project_root` directory.

### Step A: Create a virtual environment

```bash
python3 -m venv hybrid
```

### Step B: Activate the virtual environment

```bash
source hybrid/bin/activate
```

You will know it worked if your terminal prompt starts with `(hybrid)`.

### Step C: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Training the Model

Run the following command to start training:

```bash
python train.py --data_path "data/Data_Alpha_0_25" --epochs 100 --batch_size 16 --lr 0.0001
```

During training:

* A progress bar will show training and validation loss for each epoch.
* When a new best validation loss is achieved:

  * A `saved_models/` directory is created automatically, if not there.
  * The best model is saved as:

```text
saved_models/best_hybrid_Dr_H_2.pth
```

---

## 4. Evaluation and Visualization

After training, evaluate the model using:

```bash
python evaluate_and_visualize.py --data_path "data/Data_Alpha_0_25" --checkpoint "saved_models/best_hybrid_Dr_H_2.pth" --batch_size 16
```

During evaluation:

* The trained model is loaded.
* Inference is performed on the test dataset.
* A `results/` directory is created automatically.

The following outputs are generated:

* `metrics_phase2_hybrid_Dr_H.txt`
  Contains RMSE, MAE, and R² scores.

* `ladder_phase2_hybrid_Dr.png`
  Predicted vs actual plot for Dr.

* `ladder_phase2_hybrid_H.png`
  Predicted vs actual plot for H.

---

## 5. Shutting Down

To exit the virtual environment:

```bash
deactivate
```

---
