import os
import sys
import torch

# --- Project import path ---
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.predictors.loss import Poly1CrossEntropyLoss
from src.trainers.train_transformer import train
from src.models.encoders.transformers_timex import TransformerMVTS
from src.data_utils import process_Synth
from src.utils.predictors import eval_mvts_transformer

from src.datagen.spikes_data_new import SpikeTrainDataset as _SpikeTrainDataset
sys.modules['__main__'].SpikeTrainDataset = _SpikeTrainDataset

# -----------------------------
# All params in one place
# -----------------------------
PARAMS = {
    # data
    "dataset_name": "seqcombmv",                  # used for folder naming
    "base_path": "/home/graph-winit/data/SeqCombMV",
    "n_classes": 4,
    "splits": 5,                                  # run splits 1..splits

    # training
    "batch_size": 64,
    "epochs": 300,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "use_scheduler": False,
    "show_sizes": False,

    # model (TransformerMVTS)
    "nlayers": 2,
    "nhead": 1,
    "trans_dim_feedforward": 64,
    "trans_dropout": 0.25,
    "d_pe": 16,

    # loss
    "poly_epsilon": 1.0,  # Poly-1 CE extra term
}

# -----------------------------
# Utility: ckpt path builder
# -----------------------------
def make_ckpt_dir_and_names(root_dir, model_family, dataset_name, params, split):
    """
    root_dir: <project_root>/ckpt
    model_family: 'transformer' | 'gru1layer' | ...
    dataset_name: e.g. 'seqcombmv'
    params: PARAMS dict
    split: int
    """
    ckpt_dir = os.path.join(root_dir, model_family, dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Include distinguishing hyperparams in filename (like your GRU example)
    fname = (
        f"model_bs_{params['batch_size']}"
        f"_layers_{params['nlayers']}"
        f"_head_{params['nhead']}"
        f"_ff_{params['trans_dim_feedforward']}"
        f"_drop_{params['trans_dropout']}"
        f"_split_{split}.pt"
    )
    path_gpu = os.path.join(ckpt_dir, fname)
    path_cpu = os.path.join(ckpt_dir, fname.replace(".pt", "_cpu.pt"))
    return path_gpu, path_cpu

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss
    clf_criterion = Poly1CrossEntropyLoss(
        num_classes=PARAMS["n_classes"],
        epsilon=PARAMS["poly_epsilon"],
        weight=None,
        reduction='mean'
    )

    # Paths
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CKPT_ROOT = os.path.join(ROOT, "ckpt")

    # Splits loop
    for i in range(1, PARAMS["splits"] + 1):
        D = process_Synth(split_no=i, device=device, base_path=PARAMS["base_path"])
        train_loader = torch.utils.data.DataLoader(
            D['train_loader'],
            batch_size=PARAMS["batch_size"],
            shuffle=True
        )
        val, test = D['val'], D['test']

        # Build model (d_inp=D, max_len=T from val tuple)
        model = TransformerMVTS(
            d_inp=val[0].shape[-1],
            max_len=val[0].shape[0],
            n_classes=PARAMS["n_classes"],
            nlayers=PARAMS["nlayers"],
            nhead=PARAMS["nhead"],
            trans_dim_feedforward=PARAMS["trans_dim_feedforward"],
            trans_dropout=PARAMS["trans_dropout"],
            d_pe=PARAMS["d_pe"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=PARAMS["lr"],
            weight_decay=PARAMS["weight_decay"]
        )

        spath, spath_cpu = make_ckpt_dir_and_names(
            CKPT_ROOT, "transformer", PARAMS["dataset_name"], PARAMS, i
        )

        model, loss, auc = train(
            model,
            train_loader,
            val_tuple=val,
            n_classes=PARAMS["n_classes"],
            num_epochs=PARAMS["epochs"],
            save_path=spath,                 # trainer saves best here
            optimizer=optimizer,
            show_sizes=PARAMS["show_sizes"],
            use_scheduler=PARAMS["use_scheduler"],
        )

        # Also save a CPU copy for portability
        model_sdict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(model_sdict_cpu, spath_cpu)     # <-- no .format() bug

        # Final test eval
        f1 = eval_mvts_transformer(test, model)
        print(f"[split {i}] Test F1: {f1:.4f}")

if __name__ == "__main__":
    main()
