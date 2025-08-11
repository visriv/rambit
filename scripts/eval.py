# occlusion_eval.py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Union

# ------------------------------------------------------------
# Utility: model → probabilities for AUROC (binary or multiclass)
# ------------------------------------------------------------
@torch.no_grad()
def _probs(model: torch.nn.Module, X: torch.Tensor) -> np.ndarray:
    """
    Returns class probabilities:
      - If model outputs shape (N, C): softmax
      - If shape (N,) or (N,1): sigmoid positive-class prob
    """
    logits = model(X)
    if logits.ndim == 1 or logits.shape[-1] == 1:
        p = torch.sigmoid(logits.reshape(-1))
        return torch.stack([1 - p, p], dim=1).cpu().numpy()
    else:
        return torch.softmax(logits, dim=-1).cpu().numpy()

# ------------------------------------------------------------
# Occluders
# ------------------------------------------------------------
def occlude_bottom_k(X: torch.Tensor, S: np.ndarray, k: float, strategy="zero") -> torch.Tensor:
    """Bottom-k% features (least salient). S is (N,T,F) importance. X is (N,T,F)."""
    X_occ = X.clone()
    N, T, F = X.shape
    total = T * F
    n = int(k * total)
    for i in range(N):
        idx = np.argsort(S[i].ravel())[:n]       # bottom-k
        t, f = np.unravel_index(idx, (T, F))
        if strategy == "zero":
            X_occ[i, t, f] = 0.0
        elif strategy == "mean":
            X_occ[i, t, f] = X[i].mean()
        else:
            raise ValueError("strategy must be 'zero' or 'mean'")
    return X_occ

def occlude_top_k(X: torch.Tensor, S: np.ndarray, k: float, strategy="zero") -> torch.Tensor:
    """Top-k% features (most salient)."""
    X_occ = X.clone()
    N, T, F = X.shape
    total = T * F
    n = int(k * total)
    for i in range(N):
        idx = np.argsort(-S[i].ravel())[:n]      # top-k
        t, f = np.unravel_index(idx, (T, F))
        if strategy == "zero":
            X_occ[i, t, f] = 0.0
        elif strategy == "mean":
            X_occ[i, t, f] = X[i].mean()
        else:
            raise ValueError("strategy must be 'zero' or 'mean'")
    return X_occ

# ------------------------------------------------------------
# Figure 3: bottom-k curves (with shading via repeats or bootstraps)
# ------------------------------------------------------------
def evaluate_bottomk_curves(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    method_scores: Dict[str, np.ndarray],   # method → (N,T,F) saliency
    k_values: List[float],
    device: torch.device,
    repeats: int = 1,                        # >1 adds variance for shading
    strategy: str = "zero",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns dict: method -> (mean_auc_per_k, std_auc_per_k)
    """
    model.eval()
    X = X.to(device)
    y_np = y.cpu().numpy()
    results = {}
    for name, S in method_scores.items():
        auc_means, auc_stds = [], []
        for k in k_values:
            aucs = []
            for _ in range(repeats):
                X_occ = occlude_bottom_k(X, S, k, strategy=strategy).to(device)
                p = _probs(model, X_occ)
                # binary vs multiclass AUROC
                auc = roc_auc_score(y_np, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(
                    y_np, p, multi_class="ovr", average="macro"
                )
                aucs.append(auc)
            auc_means.append(np.mean(aucs))
            auc_stds.append(np.std(aucs))
        results[name] = (np.array(auc_means), np.array(auc_stds))
    return results

def plot_bottomk_curves(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                        k_values: List[float], title: str):
    colors = {"Random":"#1f77b4", "Dynamask":"#ff7f0e", "Timex":"#2ca02c", "Ours":"#d62728"}
    plt.figure(figsize=(5,4))
    for method, (m, s) in results.items():
        c = colors.get(method, None)
        plt.plot(k_values, m, marker='o', label=method, color=c)
        plt.fill_between(k_values, m - s, m + s, alpha=0.2, color=c)
    plt.xlabel("Bottom Proportion Perturbed")
    plt.ylabel("Prediction AUROC")
    plt.title(title)
    plt.ylim(0.55, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Table 4: top-10% masking with Mean/Zero substitution
# ------------------------------------------------------------
def eval_table_top10(
    model: torch.nn.Module,
    datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor]],   # name -> (X_test, y_test)
    saliency: Dict[str, Dict[str, Union[np.ndarray, List[np.ndarray]]]], 
    # saliency[dataset][method] -> (N,T,F) or list of (N,T,F) for multiple folds
    device: torch.device,
    k: float = 0.10,
    substitutions: Tuple[str, ...] = ("mean", "zero"),
) -> pd.DataFrame:
    """
    Returns a DataFrame like Table 4:
      columns: Method | Substitution | <dataset1> | <dataset2> | ... | Rank
      values:  'mean±std' strings per dataset; Rank = average rank across datasets (higher=better rank=1)
    """
    model.eval()
    # ensure consistent method ordering (Random, Dynamask, Timex, Ours if present)
    method_order = ["Random", "Dynamask", "Timex", "Timex++", "Ours"]
    present = set(m for d in saliency.values() for m in d.keys())
    methods = [m for m in method_order if m in present] + sorted(present - set(method_order))

    # compute AUROC per dataset/method/substitution (mean over folds)
    table_rows = []
    per_dataset_scores = {ds: {} for ds in datasets.keys()}  # for ranking

    for method in methods:
        for sub in substitutions:
            row = {"Method": method, "Substitution": sub}
            for ds_name, (X, y) in datasets.items():
                X = X.to(device); y_np = y.cpu().numpy()
                S_entry = saliency[ds_name][method]  # (N,T,F) or [folds]
                folds = S_entry if isinstance(S_entry, list) else [S_entry]
                aucs = []
                for S in folds:
                    X_occ = occlude_top_k(X, S, k, strategy=sub).to(device)
                    p = _probs(model, X_occ)
                    auc = roc_auc_score(y_np, p[:,1]) if p.shape[1]==2 else roc_auc_score(
                        y_np, p, multi_class="ovr", average="macro"
                    )
                    aucs.append(auc)
                mean = np.mean(aucs)
                std  = np.std(aucs)
                row[ds_name] = f"{mean:.4f}±{std:.4f}"
                per_dataset_scores[ds_name].setdefault(f"{method}|{sub}", []).append(mean)
            table_rows.append(row)

    # compute average rank per (method,sub) across datasets
    ranks = {}
    for ds_name in datasets.keys():
        # best AUROC gets rank 1
        means = {k: np.mean(v) for k, v in per_dataset_scores[ds_name].items()}
        order = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
        for r, (key, _) in enumerate(order, start=1):
            ranks.setdefault(key, []).append(r)

    for row in table_rows:
        key = f"{row['Method']}|{row['Substitution']}"
        row["Rank"] = f"{np.mean(ranks[key]):.1f}"

    # nice column order
    cols = ["Method", "Substitution"] + list(datasets.keys()) + ["Rank"]
    df = pd.DataFrame(table_rows)[cols]
    return df

# ------------------------------------------------------------
# Example driver (fill in with your actual objects)
# ------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # EXAMPLE placeholders — replace with your real objects
    # model: trained GRU moved to device
    # X_test_ds, y_test_ds: tensors (N,T,F), (N,)
    # S_dict[dataset][method] = saliency (N,T,F)  (or list of folds)
    model = ...  # your GRU
    model.to(device).eval()

    datasets = {
        "PAM":      (X_test_pam, y_test_pam),
        "Epilepsy": (X_test_epi, y_test_epi),
        "Boiler":   (X_test_boil, y_test_boil),
        # "Wafer":  (...),  "Freezer": (...)
    }

    saliency = {
        "PAM": {
            "Random":   np.random.rand(*X_test_pam.shape),
            "Dynamask": S_pam_dynamask,         # (N,T,F)
            "Timex":    S_pam_timex,
            "Ours":     S_pam_ours,             # or "Timex++" if you prefer that label
        },
        "Epilepsy": {
            "Random":   np.random.rand(*X_test_epi.shape),
            "Dynamask": S_epi_dynamask,
            "Timex":    S_epi_timex,
            "Ours":     S_epi_ours,
        },
        "Boiler": {
            "Random":   np.random.rand(*X_test_boil.shape),
            "Dynamask": S_boil_dynamask,
            "Timex":    S_boil_timex,
            "Ours":     S_boil_ours,
        },
    }

    # ---- Figure 3-style curves for one dataset (optional) ----
    k_values = [0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]
    res = evaluate_bottomk_curves(model, *datasets["PAM"], {m:saliency["PAM"][m] for m in saliency["PAM"]},
                                  k_values, device, repeats=1, strategy="zero")
    plot_bottomk_curves(res, k_values, title="PAM")

    # ---- Table 4-style top-10% masking ----
    df = eval_table_top10(model, datasets, saliency, device, k=0.10, substitutions=("mean","zero"))
    print(df.to_string(index=False))
    # Optionally save
    # df.to_csv("table4_top10_masking.csv", index=False)
