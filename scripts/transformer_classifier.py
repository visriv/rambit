import torch
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.predictors.loss import Poly1CrossEntropyLoss
from src.trainers.train_transformer import train
from src.models.encoders.transformers_timex import TransformerMVTS
from src.data_utils import process_Synth
from src.utils.predictors import eval_mvts_transformer
from src.datagen.spikes_data_new import SpikeTrainDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clf_criterion = Poly1CrossEntropyLoss(
    num_classes = 4,
    epsilon = 1.0,
    weight = None,
    reduction = 'mean'
)
base_path = '/home/graph-winit/data/SeqCombMV'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes up from scripts/
CKPT_DIR = os.path.join(ROOT, "ckpt")

for i in range(1, 6):
    D = process_Synth(split_no = i, device = device, base_path = base_path)
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    model = TransformerMVTS(
        d_inp = val[0].shape[-1],    # D of dataset
        max_len = val[0].shape[0],   # T
        n_classes = 4,
        nlayers = 2,
        nhead = 1,
        trans_dim_feedforward = 64,
        trans_dropout = 0.25,
        d_pe = 16,
        # aggreg = 'mean',
        # norm_embedding = True
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.01)
    
    # spath = '../ckpt/Scomb_transformer_split={}.pt'.format(i)
    spath = os.path.join(CKPT_DIR, f"Scomb_transformer_split={i}.pt")
    spath_cpu = os.path.join(CKPT_DIR, f"Scomb_transformer_split={i}_cpu.pt")

    model, loss, auc = train(
        model,
        train_loader,
        val_tuple = val, 
        n_classes = 4,
        num_epochs = 300,
        save_path = spath,
        optimizer = optimizer,
        show_sizes = False,
        use_scheduler = False,
    )
    
    model_sdict_cpu = {k:v.cpu() for k, v in  model.state_dict().items()}
    torch.save(model_sdict_cpu, spath_cpu.format(i))

    f1 = eval_mvts_transformer(test, model)
    print('Test F1: {:.4f}'.format(f1))