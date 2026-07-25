import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from TCN_8in import DataManager, BaseTCNRegressor, seed_everything, MODEL_MODE

DATA_DIR = "/Users/karan/Downloads/CellMigration_Project/DATA_TEST"  # <-- edit 
EPS = 1e-6


if MODEL_MODE == "A":
    CKPT_PATH = "tcn_8in_H2.pth"      # predicting Dr, H2 fed in as known input
    PREDICTED_PARAM = "Dr"
elif MODEL_MODE == "B":
    CKPT_PATH = "tcn_8in_Dr.pth"      # predicting H2, Dr fed in as known input
    PREDICTED_PARAM = "H2"
else:
    raise ValueError(f"Unknown MODEL_MODE: {MODEL_MODE}")


def train_if_needed(model, loaders, device, ckpt_path=CKPT_PATH, max_epochs=200, patience=25):
   
    if os.path.exists(ckpt_path):
        print(f"Found existing checkpoint '{ckpt_path}', skipping training.")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = torch.nn.HuberLoss()
    best_v, counter = float('inf'), 0

    print(f"Training-TCN (8 INPUTS | 1 OUTPUT: {PREDICTED_PARAM})")
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
    
    model.eval()

    # test_samples entries are (path, traj_idx, dr, h2) -- shuffle=False means this order
    # matches the order predictions come out in.
    test_dataset = loaders['test'].dataset
    dr_true = np.array([s[2] for s in test_dataset.samples], dtype=np.float64)
    h2_true = np.array([s[3] for s in test_dataset.samples], dtype=np.float64)

    all_p_norm = []
    with torch.no_grad():
        for tx, ty, tlen in loaders['test']:
            all_p_norm.append(model(tx.to(device), tlen.to(device)).cpu().numpy())

    p_norm = np.concatenate(all_p_norm)
    p_final = data_gen.target_scaler.inverse_transform(p_norm).flatten()

    if MODEL_MODE == "A":
        # Target was log10(Dr + 1) -> invert #trick
        dr_pred = 10 ** p_final - 1
        h2_pred = h2_true.copy()  # known input, fed in exactly -> trivially "predicted" perfectly
    else:
        # Target was raw H2 -> no inverse log transform
        h2_pred = p_final
        dr_pred = dr_true.copy()  # known input, fed in exactly

    y_true = np.stack([dr_true, h2_true], axis=1)
    y_pred = np.stack([dr_pred, h2_pred], axis=1)
    return y_true, y_pred


def build_accuracy_matrix(y_true, y_pred, col_idx):
   
    groups = defaultdict(list)

    for (dr_t, h2_t), (dr_p, h2_p) in zip(y_true, y_pred):
        key = (round(float(dr_t), 2), round(float(h2_t), 2))
        val_t = dr_t if col_idx == 0 else h2_t
        val_p = dr_p if col_idx == 0 else h2_p
        groups[key].append((val_t, val_p))

    dr_vals = sorted({k[0] for k in groups})
    h2_vals = sorted({k[1] for k in groups})

    dr_index = {v: i for i, v in enumerate(dr_vals)}
    h2_index = {v: i for i, v in enumerate(h2_vals)}

    acc_matrix = np.full((len(h2_vals), len(dr_vals)), np.nan)

    for (dr, h2), samples in groups.items():
        arr = np.array(samples)
        v_true = arr[:, 0]
        v_pred = arr[:, 1]

        acc = 1 - np.abs(v_pred - v_true) / (np.abs(v_true) + EPS)
        acc = np.clip(acc, 0, 1).mean()

        i = h2_index[h2]
        j = dr_index[dr]
        acc_matrix[i, j] = acc * 100

    return dr_vals, h2_vals, acc_matrix


def plot_heatmap(matrix, dr_vals, h2_vals, title, colorbar_label, out_path, vmin=90, vmax=100):
    fig, ax = plt.subplots(figsize=(16, 10))

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
    ax.set_xticklabels([f"{v:.2f}" for v in dr_vals], rotation=90, fontsize=8)

    ax.set_yticks(np.arange(len(h2_vals)))
    ax.set_yticklabels([f"{v:.2f}" for v in h2_vals], fontsize=9)

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

    model = BaseTCNRegressor(input_size=8).to(device)
    model = train_if_needed(model, loaders, device)

    y_true, y_pred = compute_test_predictions(model, data_gen, loaders, device)

    col_idx = 0 if MODEL_MODE == "A" else 1  # 0=Dr, 1=H2 -- whichever is actually predicted

    overall_acc = (
        np.clip(
            1 - np.abs(y_pred[:, col_idx] - y_true[:, col_idx]) /
            (np.abs(y_true[:, col_idx]) + EPS),
            0,
            1
        ).mean() * 100
    )

    dr_vals, h2_vals, acc_matrix = build_accuracy_matrix(y_true, y_pred, col_idx)

    print(f"Grid: {len(dr_vals)} Dr values x {len(h2_vals)} H2 values")
    print(f"{PREDICTED_PARAM} Accuracy Range : {np.nanmin(acc_matrix):.2f}% - {np.nanmax(acc_matrix):.2f}%")
    print(f"Overall {PREDICTED_PARAM} Accuracy: {overall_acc:.2f}%")

    out_path = f"{PREDICTED_PARAM.lower()}_accuracy_heatmap.png"
    plot_heatmap(
        matrix=acc_matrix,
        dr_vals=dr_vals,
        h2_vals=h2_vals,
        title=(f"TCN Matrix Evaluation (70_20_10) - {PREDICTED_PARAM} Task "
               f"(8in \u2192 1out, MODE {MODEL_MODE})\nOverall Accuracy = {overall_acc:.2f}%"),
        colorbar_label=f"Relative Accuracy ({PREDICTED_PARAM}) %",
        out_path=out_path
    )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()