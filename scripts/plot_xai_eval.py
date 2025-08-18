import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------- CONFIG ----------
DATA_DIR = "/home/users/nus/e1333861/graph-winit/output/gru1layer/boiler"
PATTERN = os.path.join(DATA_DIR, "results*.csv")  # matches results.csv, results (1).csv, results(2).csv, etc
OUT_DIR = os.path.join(DATA_DIR, "plots")
METRICS = ["auc_drop", "comp_k", "suff_k", "avg_masked_count"]
DIRECTION_ORDER = ["top", "bottom"]  # will auto-fallback to whatever exists
# ----------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# Read & merge
paths = sorted(glob.glob(PATTERN))
if not paths:
    raise FileNotFoundError(f"No files found with pattern: {PATTERN}")

dfs = []
for p in paths:
    try:
        df_i = pd.read_csv(p)
        df_i["__source"] = os.path.basename(p)
        dfs.append(df_i)
    except Exception as e:
        print(f"[WARN] Failed to read {p}: {e}")

if not dfs:
    raise RuntimeError("No readable CSVs found.")

df = pd.concat(dfs, ignore_index=True)

# Basic sanity: required columns
required_cols = {"explainer", "k", "direction"} | set(METRICS)
missing = required_cols - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns in merged data: {missing}")

# Clean types
df["k"] = pd.to_numeric(df["k"], errors="coerce")
for m in METRICS:
    df[m] = pd.to_numeric(df[m], errors="coerce")
df = df.dropna(subset=["k", "explainer", "direction"])

# Shorten explainer labels: before first underscore
df["explainer_short"] = df["explainer"].astype(str).apply(lambda s: s.split("_")[0])

# ICML-ish matplotlib style (no seaborn)
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Directions to plot (keep the specified order if present)
directions_in_data = [d for d in DIRECTION_ORDER if d in df["direction"].unique()]
for d in df["direction"].unique():
    if d not in directions_in_data:
        directions_in_data.append(d)

def plot_one(metric: str, direction: str):
    # Filter one direction and required columns
    sub = df[df["direction"] == direction].copy()
    if sub.empty:
        print(f"[INFO] Skipping: direction={direction} has no rows.")
        return

    # Aggregate across repeats: mean ± std for each (explainer_short, k)
    agg = (sub.groupby(["explainer_short", "k"], as_index=False)[metric]
              .agg(mean="mean", std="std", n="count")
              .sort_values(["explainer_short", "k"]))

    # Create plot
    fig, ax = plt.subplots()

    # Plot each explainer curve
    for explainer, g in agg.groupby("explainer_short"):
        x = g["k"].to_numpy()
        y = g["mean"].to_numpy()
        s = g["std"].to_numpy()

        # Sort by k just in case
        order = np.argsort(x)
        x, y, s = x[order], y[order], s[order]

        ax.plot(x, y, marker="o", linewidth=2, markersize=5, label=str(explainer))
        # Shaded std if there are multiple runs
        if np.isfinite(s).any() and (g["n"] > 1).any():
            ax.fill_between(x, y - s, y + s, alpha=0.15)

    # Labels & legend
    ax.set_xlabel("k")
    pretty_y = {"auc_drop": "AUC drop (↓)",
                "comp_k": "Comp K",
                "suff_k": "Suff K",
                "avg_masked_count": "Avg Masked Count"}.get(metric, metric)
    ax.set_ylabel(pretty_y)

    # Nice integer ticks for k
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

    # Place legend outside
    ax.legend(title="Explainer", frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Optional title
    ax.set_title(f"{metric} vs k  ·  direction={direction}")

    fig.tight_layout()

    # Save + show
    out_name = f"{metric}_vs_k__direction_{direction}.png"
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out_path}")
    plt.show()
    plt.close(fig)

# Generate all 8 plots (or fewer if some directions absent)
for direction in directions_in_data:
    for metric in METRICS:
        plot_one(metric, direction)
