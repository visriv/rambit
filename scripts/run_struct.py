from pathlib import Path
import sys, os
# Assumes struct.py lives in project-root/scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)
from src.analysis.interactions import compute_pairwise_interactions
import pathlib
from src.explainer.biwinitexplainers import BiWinITExplainer
from src.dataloader import Mimic, Boiler
from src.models.base_models import StateClassifier
import matplotlib.pyplot as plt
import numpy as np
import torch

def _resolve_path(base_path: pathlib.Path, model_type: str, num_layers: int):
    if model_type == "GRU":
        return base_path / f"gru{num_layers}layer"
    elif model_type == "LSTM":
        return base_path / "lstm"
    elif model_type == "CONV":
        return base_path / "conv"
    else:
        raise Exception("Unknown model type ({})".format(model_type))
        
def _model_name() -> str:
    shortened_args = {
        "bs": 32,
        "hid": 200,
        "drop": 0.5,
    }

    num_layers = 1
    if num_layers is not None and num_layers != 1:
        shortened_args["lay"] = num_layers
    rnn_type = 'gru'
    str_list = ["model"]
    if rnn_type is not None and rnn_type != "gru":
        str_list.append(rnn_type)
    str_list.extend([f"{key}_{value}" for key, value in shortened_args.items()])
    return "_".join(str_list)


batch_size = 32
data_seed = 1234
cv_to_use = 0
nondeterministic = False
train_ratio = 0.99
dataset_params = {"batch_size": batch_size,
                  "seed": data_seed, 
                   "cv_to_use": cv_to_use,
                   "deterministic": not nondeterministic,
                    "data_path": "./data/", 
                    "testbs": 32}

mask_strategy = "upper_triangular"
height = 3
xplainer_params = {"mask_strategy": mask_strategy,
                   "height": height
                  }

dataset = Boiler(data_path=dataset_params['data_path'],
                 batch_size = dataset_params['batch_size'],
                 testbs = dataset_params['testbs'],
                 deterministic = dataset_params['deterministic'],
                 cv_to_use = dataset_params['cv_to_use'],
                 seed = dataset_params['seed'],
                 )

dataset.load_data(train_ratio=train_ratio)

cv = 0
ckptpath = "./ckpt/"   
model_type = 'GRU'
base_ckpt_path = pathlib.Path(ckptpath)
num_layers = 1
ckpt_path = _resolve_path(base_ckpt_path, model_type, num_layers)
device = 'cuda'



# 0) Prepare your model
model = StateClassifier(
                feature_size=dataset.feature_size,
                num_states=dataset.num_classes,
                hidden_size=200,
                device=device,
                rnn=model_type,
                num_layers=num_layers,
                dropout=0.5,
            )


# 1) Prepare your explainer and data
explainer = BiWinITExplainer(
                    device,
                    dataset.feature_size,
                    dataset.get_name(),
                    path= ckpt_path / dataset.get_name() / str(cv),
                    # train_loader=dataset.train_loaders,  # for CF generaton if required, NOT for training 
                    other_args = xplainer_params
                )   


model_path = ckpt_path / dataset.get_name()
_model_file_name = model_path / f"{_model_name()}_{cv}.pt"

# ckpt_path = "path/to/your/checkpoint.pt"   # or .pth, .tar, etc.
ckpt = torch.load(str(_model_file_name), map_location=torch.device(device))
state_dict = ckpt.get("state_dict", ckpt)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
explainer.set_model(model,
                    set_eval=True)


# your BaseExplainer subclass, already .fit() or loaded
test_iterator = iter(dataset.test_loader)
x_batch, y_batch = next(test_iterator)  # (batch_size, features, timesteps), (batch_size,)

X_test = x_batch.to(device)
print(f"✔️  Fetched one batch: x_batch.shape = {X_test.shape}, y_batch.shape = {y_batch.shape}")

pixels     = [(10,0), (10,1), (10,2), (19,0), (19,1), (19,2)]  
save_root  = Path("outputs")
run_id     = "2025-07-15_experiment1"

# 2) Run the interaction study
delta = compute_pairwise_interactions(
    explainer=explainer,
    X=X_test,
    pixels=pixels,
    save_dir=save_root,
    run_id=run_id
)


# 3) Plot

# Paths (ensure these match your script’s variables)
# save_root = Path("outputs")
# run_id = "2025-07-15_experiment1"
inter_dir = save_root / run_id / "interactions"

# 1) load the raw delta array
delta_raw = np.load(inter_dir / "delta_I_all.npy")  # shape (n, n)
n, _, B = delta_raw.shape


# 2) compute mean + std over the examples axis
delta_mean = delta_raw.mean(axis=2)              # shape: (n, n)
delta_std  = delta_raw.std(axis=2)               # shape: (n, n)


# 3) extract upper-triangle entries and labels
pair_labels = []
mean_vals   = []
std_vals    = []
for i in range(n):
    for j in range(i+1, n):
        pair_labels.append(f"{pixels[i]}–{pixels[j]}")
        mean_vals.append(delta_mean[i, j])
        std_vals .append(delta_std[i, j])

mean_vals = np.array(mean_vals)  # shape: (n*(n-1)/2,)
std_vals  = np.array(std_vals)

# 4) summary
print(f"Mean ΔI over pixel-pair means: {mean_vals.mean():.4f}")
print(f"Std  ΔI over pixel-pair means: {mean_vals.std():.4f}")

# 5) histogram of **all** raw ΔI values (flattened)
plt.figure(figsize=(6,4))
plt.hist(delta_raw.flatten(), bins=30, edgecolor="k")
plt.title("Histogram of ALL ΔI values")
plt.xlabel("ΔI")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(inter_dir/"hist_all_delta.png")
plt.close()

# 6) histogram of per-pair means
plt.figure(figsize=(6,4))
plt.bar(range(len(mean_vals)), mean_vals, yerr=std_vals, capsize=3)
plt.xticks(range(len(mean_vals)), pair_labels, rotation=90)
plt.title("Mean ΔI per Pixel-Pair (with ±1 std)")
plt.ylabel("Mean ΔI")
plt.tight_layout()
plt.savefig(inter_dir/"bar_mean_delta.png")
plt.close()

# 7) optional: plot distributions of a few selected pairs
#    e.g. for the first 3 pairs
for idx in range(min(3, len(pair_labels))):
    plt.figure(figsize=(6,3))
    plt.hist(delta_raw.reshape(-1, B)[idx], bins=20, edgecolor="k")
    plt.title(f"ΔI Distribution for {pair_labels[idx]}")
    plt.xlabel("ΔI")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(inter_dir/f"hist_pair_{idx}.png")
    plt.close()

print("Plots saved under:", inter_dir)