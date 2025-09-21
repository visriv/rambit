# RaMbIT (a.k.a. JIMEx) (Work In Progress)

RaMbIT (Randomized Mask-Based Importance Testing) is a framework for **feature-time attribution in multivariate time series models**.  
It builds on the ideas from [WinIT) but introduces a randomized masking procedure to obtain fine-grained attributions.
Very similar to Shapley Value Sampling
Here we study the effect of different kinds of mask/coalitions on the attribution scores.


---

## 🔍 What it does
- Computes an **importance matrix** of shape **D × T** (features × timesteps) for each sequence in a batch.
- Attribution is based on comparing model predictions between two masked versions:
  - **M1**: base mask with target cell flipped (replaced by counterfactual)
  - **M2**: base mask with target cell kept
- The difference in prediction fidelity gives the **importance of that cell**.
- Aggregates across:
  - `L` random masks
  - `W` window shifts
  - `S` counterfactual samples

---

## ⚙️ Complexity
For a batch of size **B**, features **D**, timesteps **T**:
- Counterfactual generation: `O(D × S × B × T)`
- Model forward passes: `O(L × W × S × f(B,D,T))`
- Final attribution: **importance scores for every (d,t) cell**.

---

## 📦 Installation
Clone this repo and install dependencies:

```bash
git clone https://github.com/<your-username>/rambit.git
cd rambit
pip install -r requirements.txt
```

---

## 🚀 Usage

### Attribution API

```python
from rambit import RaMbIT

# X: (B, D, T) input batch
attributor = RaMbIT(model, num_samples=10, Wt_max=10, Wd_max=5, window_size=5, L=20)
I_all = attributor.attribute(X)   # (B, D, T) importance map
```

### Key Parameters
- `L`: number of random masks
- `W`: window size for temporal shifts
- `S`: number of counterfactual samples
- `Wt_max, Wd_max`: max temporal & feature window sizes
- `all_zero_cf`: if True, uses all-zero counterfactuals instead of sampling

---

## 📊 Metrics
Following Dynamask definitions:
- **Mask Information** \( I_M(A) \)
- **Mask Entropy** \( S_M(A) \)

RaMbIT supports computing these **with ground-truth salient sets (A)** on synthetic data, or **without A** on real data.

---

## 📂 Repository structure
```
rambit/
├── src/
│   ├── trainers/           # training utilities
│   ├── models/             # model definitions (GRU, CNN, LSTM, Transformer)
│   ├── utils/              # attribution, masking, metrics
│   └── data_utils/         # synthetic & real dataset loaders
├── scripts/
│   ├── transformer_classifier.py
│   └── ...
├── ckpt/                   # checkpoints
└── README.md
```

---

## 📝 Citation
If you use this work, please cite:

```
@article{your2025rambit,
  title={RaMbIT: Randomized Mask-Based Importance Testing for Time Series Attribution},
  author={Your Name},
  year={2025},
  journal={GitHub Repository}
}
```

---
