"""
TRA (Total Relative Accuracy) heatmap for the TCN regressor.
TRA = RelAcc(Dr) x RelAcc(H2), per (Dr, H2) parameter combo, in %.
Assumes TCN_Base.py is in the same directory (imports DataManager, BaseTCNRegressor).
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from TCN_Base import DataManager, BaseTCNRegressor, seed_everything

CKPT_PATH = "base_tcn_7in.pth"
DATA_DIR = "/Users/karan/Downloads/CellMigration_Project/DATA_TEST"  # <-- edit if needed--- definately needed
EPS = 1e-6


def train_if_needed(model, loaders, device, ckpt_path=CKPT_PATH, max_epochs=200, patience=25):
    """Train only if no checkpoint exists yet. Mirrors run_pipeline()'s training loop exactly."""
    if os.path.exists(ckpt_path):
        print(f"Found existing checkpoint '{ckpt_path}', skipping training.")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = torch.nn.HuberLoss()
    best_v, counter = float('inf'), 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        t_loss, n_train = 0.0, 0
        for x, y, lens in loaders['train']:
            x, y, lens = x.to(device), y.to(device), lens.to(device)
            optimizer.zero_grad()
            p = model(x, lens)
            loss = criterion(p, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x.size(0)
            n_train += x.size(0)

        model.eval()
        v_loss, n_val = 0.0, 0
        with torch.no_grad():
            for vx, vy, vlens in loaders['val']:
                vx, vy, vlens = vx.to(device), vy.to(device), vlens.to(device)
                v_loss += criterion(model(vx, vlens), vy).item() * vx.size(0)
                n_val += vx.size(0)

        avg_v = v_loss / n_val
        scheduler.step(avg_v)

        if epoch % 5 == 0:
            print(f"Ep {epoch:03d} | Train: {t_loss/n_train:.4f} | Val: {avg_v:.4f}")

        if avg_v < best_v:
            best_v, counter = avg_v, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


def compute_test_predictions(model, data_gen, loaders, device):
    """
    Run inference on the test set and return per-sample:
    (dr_true, h2_true, dr_pred, h2_pred) in ORIGINAL (unscaled) units,
    in the exact iteration order of loaders['test'] (shuffle=False -> matches test_samples order).
    """
    model.eval()
    all_p_norm, all_y_norm = [], []
    with torch.no_grad():
        for tx, ty, tlen in loaders['test']:
            all_p_norm.append(model(tx.to(device), tlen.to(device)).cpu().numpy())
            all_y_norm.append(ty.numpy())

    p_final = data_gen.target_scaler.inverse_transform(np.concatenate(all_p_norm))
    y_final = data_gen.target_scaler.inverse_transform(np.concatenate(all_y_norm))


    p_final[:, 0] = 10 ** p_final[:, 0] - 1
    y_final[:, 0] = 10 ** y_final[:, 0] - 1

    return y_final, p_final  


def build_accuracy_matrices(y_true, y_pred):
    """
    Computes:
        Dr Relative Accuracy matrix
        H2 Relative Accuracy matrix
        TRA = DrAcc * H2Acc

    Returns:
        dr_vals
        h2_vals
        dr_matrix
        h2_matrix
        tra_matrix
    """

    groups = defaultdict(list)

    for (dr_t, h2_t), (dr_p, h2_p) in zip(y_true, y_pred):
        key = (round(float(dr_t), 2), round(float(h2_t), 2))
        groups[key].append((dr_t, h2_t, dr_p, h2_p))

    dr_vals = sorted({k[0] for k in groups})
    h2_vals = sorted({k[1] for k in groups})

    dr_index = {v: i for i, v in enumerate(dr_vals)}
    h2_index = {v: i for i, v in enumerate(h2_vals)}

    dr_matrix = np.full((len(h2_vals), len(dr_vals)), np.nan)
    h2_matrix = np.full((len(h2_vals), len(dr_vals)), np.nan)
    tra_matrix = np.full((len(h2_vals), len(dr_vals)), np.nan)

    for (dr, h2), samples in groups.items():

        arr = np.array(samples)

        dr_true = arr[:,0]
        h2_true = arr[:,1]

        dr_pred = arr[:,2]
        h2_pred = arr[:,3]

        dr_acc = 1 - np.abs(dr_pred-dr_true)/(np.abs(dr_true)+EPS)
        h2_acc = 1 - np.abs(h2_pred-h2_true)/(np.abs(h2_true)+EPS)

        dr_acc = np.clip(dr_acc,0,1).mean()
        h2_acc = np.clip(h2_acc,0,1).mean()

        i = h2_index[h2]
        j = dr_index[dr]

        dr_matrix[i,j] = dr_acc*100
        h2_matrix[i,j] = h2_acc*100
        tra_matrix[i,j] = dr_acc*h2_acc*100

    return dr_vals, h2_vals, dr_matrix, h2_matrix, tra_matrix


def plot_heatmap(matrix,
                 dr_vals,
                 h2_vals,
                 title,
                 colorbar_label,
                 out_path,
                 vmin=90,
                 vmax=100):

    fig, ax = plt.subplots(figsize=(16,10))

    im = ax.imshow(
        matrix,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )

    ax.set_xticks(np.arange(len(dr_vals)))
    ax.set_xticklabels(
        [f"{v:.2f}" for v in dr_vals],
        rotation=90,
        fontsize=8
    )

    ax.set_yticks(np.arange(len(h2_vals)))
    ax.set_yticklabels(
        [f"{v:.2f}" for v in h2_vals],
        fontsize=9
    )

    ax.set_xlabel(r"$Dr$", fontsize=14)
    ax.set_ylabel(r"$H_2$", fontsize=14)

    ax.set_title(title, fontsize=15)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    seed_everything(42)
    device = torch.device("mps" if torch.backends.mps.is_available()
                           else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_gen = DataManager(base_path=DATA_DIR, batch_size=16)
    loaders = data_gen.get_loaders()

    model = BaseTCNRegressor(input_size=7).to(device)
    model = train_if_needed(model, loaders, device)

    y_true, y_pred = compute_test_predictions(model, data_gen, loaders, device)
    dr_vals, h2_vals, dr_acc, h2_acc, tra = build_accuracy_matrices(
    y_true,
    y_pred
)
    print(f"Grid: {len(dr_vals)} Dr values x {len(h2_vals)} H2 values")
    print(f"Dr Accuracy Range : {np.nanmin(dr_acc):.2f}% - {np.nanmax(dr_acc):.2f}%")
    print(f"H2 Accuracy Range : {np.nanmin(h2_acc):.2f}% - {np.nanmax(h2_acc):.2f}%")
    print(f"TRA Range         : {np.nanmin(tra):.2f}% - {np.nanmax(tra):.2f}%")
    # Dr Accuracy Heatmap
    plot_heatmap(
    matrix=dr_acc,
    dr_vals=dr_vals,
    h2_vals=h2_vals,
    title="TCN Matrix Evaluation (70_20_10) - Dr Task\nRelative Accuracy (Dr)",
    colorbar_label="Relative Accuracy (Dr) %",
    out_path="dr_accuracy_heatmap.png"
)
    # H2 Accuracy Heatmap
    plot_heatmap(
    matrix=h2_acc,
    dr_vals=dr_vals,
    h2_vals=h2_vals,
    title="TCN Matrix Evaluation (70_20_10) - H₂ Task\nRelative Accuracy (H₂)",
    colorbar_label="Relative Accuracy (H₂) %",
    out_path="h2_accuracy_heatmap.png"
)
    # TRA Heatmap
    plot_heatmap(
    matrix=tra,
    dr_vals=dr_vals,
    h2_vals=h2_vals,
    title="TCN Matrix Evaluation (70_20_10) - Joint Task\nTRA = Rel.Acc(H₂) × Rel.Acc(Dr)",
    colorbar_label="Total Relative Accuracy (TRA) %",
    out_path="tra_heatmap.png"
)
    print("\nSaved:")
    print("  dr_accuracy_heatmap.png")
    print("  h2_accuracy_heatmap.png")
    print("  tra_heatmap.png")


if __name__ == "__main__":
    main()
