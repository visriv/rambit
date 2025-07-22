import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_interaction_heatmap(
    delta_I: np.ndarray,
    pixels: List[Tuple[int,int]],
    save_path: Path
):
    """
    Given an (n×n) ΔI matrix, draw a heatmap with pixel‐indices on axes.
    """
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(delta_I, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(pixels)))
    ax.set_yticks(range(len(pixels)))
    ax.set_xticklabels([f"{p}" for p in pixels], rotation=90)
    ax.set_yticklabels([f"{p}" for p in pixels])
    fig.colorbar(im, ax=ax, label="ΔI")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
