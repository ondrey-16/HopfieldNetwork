import os
import sys
from pathlib import Path

from model.hopfield_network import HopfieldNetwork
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.linalg import hadamard


def create_random_image(n: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    return np.random.choice(a=[-1, 1], size=n)

def create_number_of_random_images(n: int, seed: int, how_many: int) -> np.ndarray:
    np.random.seed(seed)
    ret = np.ndarray((how_many, n))
    for i in range(how_many):
        ret[i, :] = np.random.choice(a=[-1, 1], size=n)
    return ret

def create_pseudo_orthogonal_images(
    seed: int
) -> np.ndarray:
    # generated in sage:
    # from sage.all import hadamard_matrix
    # import numpy as np
    #
    # H = np.array(hadamard_matrix(24))
    # print(H)

    hadamard_24 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1,
         -1, 1],
        [1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1,
         1, 1],
        [1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1,
         -1],
        [1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1,
         -1],
        [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1,
         -1, 1],
        [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1,
         -1],
        [1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1,
         -1, 1],
        [1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1,
         1, 1],
        [1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1,
         -1],
        [1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1,
         -1],
        [1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1,
         -1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1],
        [1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
         -1],
        [1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1,
         -1],
        [1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1,
         -1, 1],
        [1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1,
         1, 1],
        [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1,
         -1],
        [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1,
         1, 1],
        [1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1,
         -1],
        [1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1,
         -1],
        [1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1,
         -1, 1],
        [1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1,
         -1, 1],
        [1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1,
         1, 1]
    ])
    pseudo_orthogonal_25 = np.ndarray((24, 25))
    np.random.seed(seed)
    randoms = np.random.choice(a=[-1, 1], size=24)
    for i in range (24):
        pseudo_orthogonal_25[i, :] = np.append(hadamard_24[i, :], randoms[i])

    return pseudo_orthogonal_25

def read_data_train_hebb(dataset_name: str, is_async: bool) -> tuple[HopfieldNetwork, np.ndarray]:
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

    if project_root not in sys.path:
        sys.path.append(project_root)

    patterns_path = Path(project_root) / "data" /dataset_name
    df = pd.read_csv(patterns_path, header=None)
    patterns = np.asarray(df, dtype=float)

    n = patterns.shape[1]

    hn = HopfieldNetwork(n=n, method="hebb", if_asynchronic=is_async)
    hn.train(patterns=patterns)
    stable_patterns = 0
    for i in range(patterns.shape[0]):
        if hn.check_if_stable(patterns[i]):
            stable_patterns += 1

    # print(f"Stable patterns: {stable_patterns}/{patterns.shape[0]}")
    return hn, patterns

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_energy_landscape(W: np.ndarray, patterns: np.ndarray, history: list = None, resolution: int = 80) -> None:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(patterns)

    margin = 1.5
    x_min, x_max = coords[:, 0].min() - margin, coords[:, 0].max() + margin
    y_min, y_max = coords[:, 1].min() - margin, coords[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    approx_states = pca.inverse_transform(grid_2d)
    binary_states = np.where(approx_states >= 0, 1.0, -1.0)

    energies = -0.5 * np.sum((binary_states @ W) * binary_states, axis=1)
    zz = energies.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, zz, levels=30, cmap="viridis")
    ax.contour(xx, yy, zz, levels=15, colors="black", alpha=0.25, linewidths=0.5)
    plt.colorbar(contour, ax=ax, label="Energia E(s)")

    ax.scatter(coords[:, 0], coords[:, 1], c="red", s=90, edgecolors="white", zorder=5, label="Wzorce uczące")
    for idx, (x, y) in enumerate(coords):
        ax.annotate(f"P{idx}", (x, y), xytext=(5, 5), textcoords="offset points", color="white", fontweight="bold")

    if history is not None and len(history) > 0:
        hist_coords = pca.transform(np.array(history))
        ax.plot(hist_coords[:, 0], hist_coords[:, 1], color="orange", linewidth=2, marker="o", markersize=4, zorder=6, label="Trajektoria zbiegania")
        ax.scatter(hist_coords[0, 0], hist_coords[0, 1], color="cyan", s=100, marker="X", zorder=7, label="Start (szum)")

    ax.set_title("Krajobraz energii sieci Hopfielda (rzut PCA 2D)")
    ax.set_xlabel("Główna składowa 1 (PC1)")
    ax.set_ylabel("Główna składowa 2 (PC2)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def classify_recalls(patterns: np.ndarray, recalled_patterns: np.ndarray, plot_dim: tuple, threshold: float = 0.98) -> dict:
    n_neurons = patterns.shape[1]

    stats = {"original": 0, "negative": 0, "spurious": 0}
    labels = []
    unique_map = {}
    total = recalled_patterns.shape[0]

    for state in recalled_patterns:
        overlaps = (patterns @ state) / n_neurons
        max_ov = np.max(overlaps)
        min_ov = np.min(overlaps)
        best_abs_ov = max(abs(max_ov), abs(min_ov))
        match_pct = round(((best_abs_ov + 1.0) / 2.0) * 100.0, 1)

        if max_ov >= threshold:
            label = "original"
        elif min_ov <= -threshold:
            label = "negative"
        else:
            label = "spurious"

        stats[label] += 1
        labels.append(f"{label} ({match_pct}%)")

        key = state.tobytes()
        if key not in unique_map:
            unique_map[key] = {
                "pattern": state,
                "count": 0,
                "label": label,
                "match_pct": match_pct
            }
        unique_map[key]["count"] += 1

    unique_states = sorted(unique_map.values(), key=lambda x: x["count"], reverse=True)

    summary_df = pd.DataFrame([
        {
            "Kategoria": k,
            "Liczba": v,
            "Procent [%]": round(100.0 * v / total, 1)
        }
        for k, v in stats.items()
    ])
    display(summary_df)

    u_count = len(unique_states)
    cols = min(5, u_count)
    rows = math.ceil(u_count / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.1), squeeze=False)
    for idx, ax in enumerate(axes.flat):
        if idx < u_count:
            item = unique_states[idx]
            bitmap = item["pattern"].reshape(plot_dim)
            ax.imshow(bitmap, cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
            pct = round(100.0 * item["count"] / total, 1)
            ax.set_title(f"{item['label']} ({item['match_pct']}%)\n{item['count']}x ({pct}%)", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return {"stats": stats, "labels": labels, "unique_states": unique_states}

def train_hopfield_from_array_for_stability(patterns: np.ndarray) -> tuple[bool, str]:
    n = patterns.shape[1]

    hn = HopfieldNetwork(n=n, method="hebb", if_asynchronic=True)
    hn.train(patterns=patterns)
    stable_patterns = 0
    for i in range(patterns.shape[0]):
        if hn.check_if_stable(patterns[i]):
            stable_patterns += 1

    is_stable = stable_patterns == patterns.shape[0]
    return is_stable, f"Stable patterns: {stable_patterns}/{patterns.shape[0]}"

def train_hopfield_from_array(patterns: np.ndarray) -> HopfieldNetwork:
    n = patterns.shape[1]

    hn = HopfieldNetwork(n=n, method="hebb", if_asynchronic=False)
    hn.train(patterns=patterns)
    return hn

from PIL import Image

def resize_image(name: str, size: tuple[int, int] = (200, 300)) -> Image.Image:
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

    if project_root not in sys.path:
        sys.path.append(project_root)

    path = Path(project_root) / "data" /name
    with Image.open(path) as img:
        resized = img.convert("1").resize(size)
        plt.imshow(resized)
        plt.axis("off")
        plt.show()



def load_and_resize_image_dithering(name: str, size: tuple[int, int] = (200, 300)) -> np.ndarray:
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

    if project_root not in sys.path:
        sys.path.append(project_root)

    path = Path(project_root) / "data_for_big_bitmaps" /name

    with Image.open(path) as img:
        resized = img.resize(size).convert("1")
        arr = np.asarray(resized, dtype=np.int8)
        return np.where(arr == 1, 1, -1).reshape(1, -1)