import os
import numpy as np
import matplotlib.pyplot as plt


NPZ_PATH = "../results/obf_metric.npz"
METRIC_NAME = "r_perp"
ROUND = 0
SAVE_DIR = "../figures"


def load_metric(npz_path: str, metric_name: str, round_idx: int) -> np.ndarray:
    metric_key = f"{metric_name}_{round_idx}"
    data = np.load(npz_path)

    if metric_key not in data:
        raise KeyError(
            f"Key '{metric_key}' not found in {npz_path}. "
            f"Available keys: {list(data.keys())}"
        )

    arr = data[metric_key]

    if arr.ndim != 3:
        raise ValueError(
            f"Expected {metric_key} to have shape (N, L, H), but got {arr.shape}"
        )

    return arr, metric_key


def plot_r_prep(round: int):
    arr, metric_key = load_metric(NPZ_PATH, 'r_perp', ROUND)

    mean = arr.mean(axis=0)   # (L, H)
    std = arr.std(axis=0)     # (L, H)

    _, num_layers, num_heads = arr.shape
    layers = np.arange(num_layers)

    plt.figure(figsize=(12, 7))

    for h in range(num_heads):
        plt.plot(layers, mean[:, h], label=f"head {h}")
        plt.fill_between(
            layers,
            mean[:, h] - std[:, h],
            mean[:, h] + std[:, h],
            alpha=0.2,
        )

    plt.xlabel("Layer")
    plt.ylabel(metric_key)
    plt.title(f"{metric_key} by head across layers")
    plt.xticks(layers)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, f"{metric_key}_plot.png")

    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    arr, metric_key = load_metric(NPZ_PATH, METRIC_NAME, ROUND)
    save_path = os.path.join(SAVE_DIR, f"{metric_key}_plot.png")

    plot_r_prep(round=0)

    print(f"Loaded: {NPZ_PATH}")
    print(f"Metric shape: {arr.shape}")
    print(f"Saved plot to: {save_path}")


if __name__ == "__main__":
    main()