import torch

from src.utils.predictors.loss import Poly1CrossEntropyLoss
from src.trainers.train_transformer import train
from src.models.encoders.transformer_simple import TransformerMVTS
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

for i in range(1, 6):
    D = process_Synth(split_no = i, device = device, base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/SeqCombSingleBetter')
    train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 64, shuffle = True)

    val, test = D['val'], D['test']

    model = TransformerMVTS(
        d_inp = val[0].shape[-1],
        max_len = val[0].shape[0],
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
    
    spath = 'models/Scomb_transformer_split={}.pt'.format(i)

    model, loss, auc = train(
        model,
        train_loader,
        val_tuple = val, 
        n_classes = 4,
        num_epochs = 200,
        save_path = spath,
        optimizer = optimizer,
        show_sizes = False,
        use_scheduler = False,
    )
    
    model_sdict_cpu = {k:v.cpu() for k, v in  model.state_dict().items()}
    torch.save(model_sdict_cpu, 'models/Scomb_transformer_split={}_cpu.pt'.format(i))

    f1 = eval_mvts_transformer(test, model)
    print('Test F1: {:.4f}'.format(f1))