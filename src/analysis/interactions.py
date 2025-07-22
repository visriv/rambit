import numpy as np
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Any
import torch
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)
from src.explainer.explainers import BaseExplainer  # adjust import to your project structure
from typing import Any, List, Tuple, Union

# Coords = Union[Tuple[int,int], List[Tuple[int,int]]]
Coords = Union[Tuple[int,int], List[Tuple[int,int]]]

def _canon(c: Coords) -> Tuple[Tuple[int,int], ...]:
    """
    Turn either a single (t,d) or a list of them into a tuple of (t,d) tuples.
    """
    if isinstance(c, tuple):
        return (c,)
    elif isinstance(c, list):
        return tuple(c)
    else:
        raise TypeError(f"Expected tuple or list of tuples, got {type(c)}")

def _to_numpy(v) -> np.ndarray:
    """
    Safely convert whatever joint_importance returns into a 1-D numpy array.
    Handles: torch.Tensor on GPU/CPU, python scalars, lists/tuples of tensors.
    """
    if torch.is_tensor(v):
        return v.detach().cpu().numpy().reshape(-1)
    if isinstance(v, (list, tuple)):
        # convert each element, then concat
        parts = [_to_numpy(x) for x in v]
        return np.concatenate(parts).reshape(-1)
    # numpy or scalar
    arr = np.asarray(v)
    return arr.reshape(-1)


def _to_list(c: Coords) -> List[Tuple[int,int]]:
    if isinstance(c, tuple):
        return [c]
    elif isinstance(c, list):
        return c
    else:
        raise TypeError(f"Expected tuple or list of tuples, got {type(c)}")


def single_importance(
    explainer: BaseExplainer,
    x: Any,
    p: Coords
) -> float:
    """
    Single importance:
        I({p}) = f(x) − f(x with pixel p masked)
    """
    imp_score = {}
    if isinstance(p, tuple):
        return explainer.mask_and_score_windowed(x, _to_list(p))
    else:
        for item in p:
            blob = item
            imp_score[item] = explainer.mask_and_score_windowed(x, _to_list(blob))

        return imp_score

def joint_importance(
    explainer: Any,
    x: Any,
    p: Coords,
    q: Coords
) -> float:
    """
    Joint importance:
      I(S) = f(x) - f(x with all pixels in S masked)

    Here p and/or q can each be either:
      - a single (t,d) tuple
      - a list of (t,d) tuples

    We flatten both into one `coords` list before masking.
    """
    # helper to flatten p or q into a list of tuples

    coords = _to_list(p) + _to_list(q)
    blob = coords
    # now coords is a List[Tuple[int,int]] of arbitrary length
    return explainer.mask_and_score_windowed(x, blob)






def compute_joint_importance(
    pairs: List[Tuple[Coords, Coords]],
    X: List[Any],
    explainer: Any
) -> dict:
    """
    pairs: list of (source_p, target_p), where each can be a single tuple or list of tuples.
    X:     list of examples
    explainer: your BaseExplainer instance
    """
    all_Ipq = {}

    for source_p, target_p in pairs:
        src_key = _canon(source_p)
        tgt_key = _canon(target_p)
        key = (src_key, tgt_key)

        vals = []
        for x in X:
            v = joint_importance(explainer, x.unsqueeze(0), source_p, target_p)
            v = _to_numpy(v)                 # shape (W,) or scalar
            # v = np.atleast_1d(v)             # ensure 1-D
            vals.append(v)

        arr = np.stack(vals, axis=0)         # shape (B, W)
        all_Ipq[key] = arr

    return all_Ipq

def residual_interaction_effect(
    I_pq: float,
    I_p:  float,
    I_q:  float
) -> float:
    """
    Residual Interaction effect:
        ΔI(p,q) = I({p,q}) − I({p}) − I({q})
    """
    return I_pq - I_p - I_q

# def compute_joint_importance(pairs, X, explainer):
#     all_Ipq = {}
#     for p, q in pairs:
#         vals = []
#         for x in X:
#             val = joint_importance(explainer, x.unsqueeze(0), p, q)
#             vals.append(val.detach().cpu().item())
#         arr = np.array(vals, dtype=float)
#         all_Ipq[(p, q)] = arr
#         # np.save(out_dir / f"I_pair_{p}_{q}.npy", arr)

#     return all_Ipq


## OLD function
def compute_residual_interactions(
    explainer: BaseExplainer,
    X: List[Any],
    pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    save_dir: Path,
    run_id: str
):


    """
    1) Compute single importances for all unique pixels
    2) Compute joint importances for each (p, q) in pairs
    3) Build delta tensor for each pair and example: ΔI(p,q) = Ipq - Ip - Iq
    Returns:
      all_Ip:   dict mapping pixel -> [I_p(xi) for xi in X]
      all_Ipq:  dict mapping (p,q) -> [I_{p,q}(xi) for xi in X]
      delta:    dict mapping (p,q) -> [ΔI_{p,q}(xi) for xi in X]
    """

    n = len(pairs)

    out_dir = save_dir / run_id / "interactions"
    out_dir.mkdir(parents=True, exist_ok=True)


    # Unique pixels
    all_pixels = list({tup for pair in pairs for tup in pair})
    n_pixels = len(all_pixels)
    B = len(X)

    # Maps to store scores for lookup
    all_Ip = {}    # p -> np.array of shape (B,)
    all_Ipq = {}   # (p,q) -> np.array of shape (B,)
    delta  = {}    # (p,q) -> np.array of shape (B,)

    # 1) Single importances
    for p in all_pixels:
        vals = []
        for x in X:
            val = single_importance(explainer, x.unsqueeze(0), p)
            vals.append(val.detach().cpu().item())
        arr = np.array(vals, dtype=float)
        all_Ip[p] = arr
        # np.save(out_dir / f"I_single_{p[0]}_{p[1]}.npy", arr)

    # 2) Joint importances
    all_Ipq = compute_joint_importance(pairs, X, explainer)

    

    # 3) ΔI per example for each pair
    for p, q in pairs:
        Ip  = all_Ip[p]
        Iq  = all_Ip[q]
        Ipq = all_Ipq[(p, q)]
        delta_arr = Ipq - Ip - Iq
        delta[(p, q)] = delta_arr
        # np.save(out_dir / f"delta_{p}_{q}.npy", delta_arr)

    return all_Ip, all_Ipq, delta
