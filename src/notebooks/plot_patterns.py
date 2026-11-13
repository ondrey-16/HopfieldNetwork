import numpy as np
import matplotlib.pyplot as plt
import math

def plot_patterns(patterns: np.ndarray, plot_dim: tuple) -> None:
    count = patterns.shape[0]
    cols = min(5, count)
    rows = math.ceil(count / cols)

    _, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8), squeeze=False)

    for k, ax in enumerate(axes.flat):
        if k < count:
            bitmap = patterns[k].reshape(plot_dim)

            ax.imshow(bitmap, cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title(f"Pattern: {k}")

        ax.axis("off")

    plt.tight_layout()
    plt.show()