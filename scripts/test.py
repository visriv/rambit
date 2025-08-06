import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


from pathlib import Path
import sys, os

# project root = one level above scripts/
ROOT = Path(__file__).resolve().parents[1]

# Option A: keep `src` in the import path
sys.path.insert(0, str(ROOT))          # parent of src
from src.analysis.interactions import compute_residual_interactions, compute_joint_importance, single_importance
import pathlib
from src.explainer.biwinitexplainers import BiWinITExplainer 
from src.explainer.original_winitexplainers import OGWinITExplainer 
from src.dataloader import Mimic, Boiler, SimulatedState
from src.models.base_models import StateClassifier
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict
from typing import Any, List, Tuple, Union
Coords = Union[Tuple[int,int], List[Tuple[int,int]]]


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
        "bs": 100,
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
run_id     = "2025-07-17_state"
# run_id     = "2025-07-15_boiler"


# dataset = Boiler(data_path=dataset_params['data_path'],
#                  batch_size = 32,
#                  testbs = dataset_params['testbs'],
#                  deterministic = dataset_params['deterministic'],
#                  cv_to_use = dataset_params['cv_to_use'],
#                  seed = dataset_params['seed'],
#                  )

dataset = SimulatedState(data_path=dataset_params['data_path'],
                 batch_size = 100,
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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



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
                    train_loader=dataset.train_loaders[cv],  # for CF generaton if required, NOT for training 
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






# how many batches to concat
num_batches = 2# len(dataset.test_loader)

xs = []
ys = []
it = iter(dataset.test_loader)

for _ in range(num_batches):
    x_batch, y_batch = next(it)   # x_batch: [32, 3, 100], y_batch: [32]
    xs.append(x_batch)
    ys.append(y_batch)

# now concatenate along the batch dimension
X_test = torch.cat(xs, dim=0).to(device)   # [32*10, 3, 100] == [320, 3, 100]
Y_test = torch.cat(ys, dim=0)              # [320]

print(f"✔️  Fetched {num_batches} batches:")
print(f"   X_test.shape = {X_test.shape}, Y_test.shape = {Y_test.shape}")



pixels     = [(0,10), (1,10), (2,10), (0,19), (1,19), (2,19),
              (0,13), (1,13), (2,13), 
            #   (0,39), (1,39), (2,39)
            ]  
# pixels     = [(10,0), (10,1), (10,2), (19,0), (19,1), (19,2)]  
save_root  = Path("./outputs")



import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# 3) sampling strategy function
def get_pair_strategy(
    source_pixels: List[Tuple[int,int]],
    T: int,
    D: int,
    tau: int,
    strategy: str = "same_d"
) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    output_pairs = []
    for (t, d) in source_pixels:

        if strategy == "same_d":
            t2 = t + tau
            d2 = d
            if not (0 <= t2 < T):
                t2 = T-1
            output_pairs.append(((t,d), (t2, d2)))

        elif strategy == "next_d":
            t2 = t + tau
            d2 = min(d + 1, D - 1)
            output_pairs.append(((t,d), (t2, d2)))

        elif strategy == "vertical_strip":
            t2 = t + tau
            target_blob = []

            for dd in range(D):
                target_blob.append((t2,dd))
            output_pairs.append(((t,d), target_blob))

        else:
            raise ValueError(f"Unknown strategy '{strategy}'")
    
    return output_pairs

# 4) illustrate_pair
def illustrate_pair(pair: Tuple[Tuple[int,int], Tuple[int,int]], T: int, D: int):
    grid = np.ones((T, D, 3)) * 0.8
    p, q = pair
    grid[p[0], p[1]] = [1.0, 0.5, 0.0]
    grid[q[0], q[1]] = [1.0, 0.5, 0.0]
    plt.figure(figsize=(4, 4))
    plt.imshow(grid, aspect='auto')
    plt.title(f"{p} → {q}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()




B, D, T = X_test.shape

source_pixels     = [(0,0), (1,0), (2,0), 
                    (0,1), (1,1), (2,1),
                    (0,2), (1,2), (2,2), 
                    ]  
save_root  = Path("./outputs")
run_id     = "2025-07-17_state"

out_dir = save_root / run_id / "interactions"
# 8) line plots of mean ΔI vs tau for each strategy
tau_values = list(range(3, 97, 3))



print('source_pixels', source_pixels)
num_pairs = len(source_pixels)
out_dir = save_root / run_id / "interactions_window_strip"
pair_strategies =  ["vertical_strip"]#, "next_d"]
tau_values = list(range(3, 19, 3))

import os, torch
print("PID:", os.getpid())
print("current_device:", torch.cuda.current_device())
print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.memory_summary(torch.cuda.current_device()))
