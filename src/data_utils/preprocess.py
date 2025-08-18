import random
import torch
import numpy as np
import pandas as pd
import sys, os
from .utils_phy12 import *
from sklearn.model_selection import train_test_split
import warnings

base_path = '/home/owq978/TimeSeriesXAI/PAMdata/PAMAP2data/'

# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class PAMchunk:
    '''
    Class to hold chunks of PAM data
    '''
    def __init__(self, train_tensor, static, time, y, device=None):
        # Keep them on CPU
        self.X      = train_tensor        # shape: [features, n_samples, time]
        self.static = static              # or None
        self.time   = time                # shape: [n_samples, time]
        self.y      = y                   # shape: [n_samples, …]
        self.device = device

    def choose_random(self):
        n_samp = len(self.X)           
        idx = random.choice(np.arange(n_samp))
        
        static_idx = None if self.static is None else self.static[idx]
        print('In chunk', self.time.shape)
        return self.X[:,idx,:].unsqueeze(dim=1), \
            self.time[:,idx].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx

    def __getitem__(self, idx):
        # slice out the idx-th sample **on CPU** 
        X_i      = self.X[:, idx, :].unsqueeze(1)   # → [features, 1, time]
        time_i   = self.time[idx].unsqueeze(-1)     # → [time, 1]
        y_i      = self.y[idx].unsqueeze(0)         # → [1, …]
        static_i = None if self.static is None else self.static[idx]

        # now move *only this sample* to GPU
        if self.device is not None:
            X_i      = X_i.to(self.device)
            time_i   = time_i.to(self.device)
            y_i      = y_i.to(self.device)
            if static_i is not None:
                static_i = static_i.to(self.device)

        return X_i, time_i, y_i, static_i

class RWDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X = X # Shape: (T, N, d)
        self.times = times # Shape: (T, N)
        self.y = y # Shape: (N,)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        time = self.times[:,idx]
        y = self.y[idx]
        return x, time, y 

class PAMDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X = X # Shape: (T, N, d)
        self.times = times # Shape: (T, N)
        self.y = y # Shape: (N,)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y 




def process_PAM(split_no = 1, device = None, base_path = base_path, gethalf = False):
    split_path = 'splits/PAMAP2_split_{}.npy'.format(split_no)
    idx_train, idx_val, idx_test = np.load(os.path.join(base_path, split_path), allow_pickle=True)

    Pdict_list = np.load(base_path / 'processed_data' / 'PTdict_list.npy', allow_pickle=True)
    arr_outcomes = np.load(base_path / 'processed_data' / 'arr_outcomes.npy', allow_pickle=True)

    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]

    y = arr_outcomes[:, -1].reshape((-1, 1))

    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    #return Ptrain, Pval, Ptest, ytrain, yval, ytest

    T, F = Ptrain[0].shape
    D = 1

    Ptrain_tensor = Ptrain
    Ptrain_static_tensor = np.zeros((len(Ptrain), D))

    mf, stdf = getStats(Ptrain)
    Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
    Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
    Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_other(Ptest, ytest, mf, stdf)

    Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
    Pval_tensor = Pval_tensor.permute(1, 0, 2)
    Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

    if gethalf:
        Ptrain_tensor = Ptrain_tensor[:,:,:(Ptrain_tensor.shape[-1] // 2)]
        Pval_tensor = Pval_tensor[:,:,:(Pval_tensor.shape[-1] // 2)]
        Ptest_tensor = Ptest_tensor[:,:,:(Ptest_tensor.shape[-1] // 2)]

    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    train_chunk = PAMchunk(Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor, device = device)
    val_chunk = PAMchunk(Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor, device = device)
    test_chunk = PAMchunk(Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor, device = device)

    return train_chunk, val_chunk, test_chunk

def zip_x_time_y(X, time, y):
    # Break up all args into lists in first dimension:
    Xlist = [X[:,i,:].unsqueeze(1) for i in range(X.shape[1])]
    timelist = [time[:,i].unsqueeze(dim=1) for i in range(time.shape[1])]
    ylist = [y[i] for i in range(y.shape[0])]

    return list(zip(Xlist, timelist, ylist))

class ECGchunk:
    '''
    Class to hold chunks of ECG data
    '''
    def __init__(self, train_tensor, static, time, y, device = None):
        self.X = train_tensor.to(device)
        self.static = None if static is None else static.to(device)
        self.time = time.to(device)
        self.y = y.to(device)

    def choose_random(self):
        n_samp = self.X.shape[1]           
        idx = random.choice(np.arange(n_samp))

        static_idx = None if self.static is None else self.static[idx]
        #print('In chunk', self.time.shape)
        return self.X[idx,:,:].unsqueeze(dim=1), \
            self.time[:,idx].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx

    def get_all(self):
        static_idx = None # Doesn't support non-None 
        return self.X, self.time, self.y, static_idx

    def __getitem__(self, idx): 
        static_idx = None if self.static is None else self.static[idx]
        return self.X[:,idx,:], \
            self.time[:,idx], \
            self.y[idx].unsqueeze(dim=0)
            #static_idx

def mask_normalize_ECG(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    P_tensor = P_tensor.numpy()
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2,0,1)).reshape(F,-1)
    M = 1*(P_tensor>0) + 0*(P_tensor<=0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F,N,T)).transpose((1,2,0))
    #Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pnorm_tensor
    #return Pnorm_tensor

def tensorize_normalize_ECG(P, y, mf, stdf):
    F, T = P[0].shape

    P_time = np.zeros((len(P), T, 1))
    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
    P_tensor = mask_normalize_ECG(P, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0

    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor)
    return P_tensor, None, P_time, y_tensor

ecg_base_path = '/home/owq978/TimeSeriesXAI/ECGdata/ECG'
def process_ECG(split_no = 1, device = None, base_path = ecg_base_path):

    # train = torch.load(os.path.join(loc, 'train.pt'))
    # val = torch.load(os.path.join(loc, 'val.pt'))
    # test = torch.load(os.path.join(loc, 'test.pt'))

    split_path = 'split_{}.npy'.format(split_no)
    idx_train, idx_val, idx_test = np.load(os.path.join(base_path, split_path), allow_pickle = True)

    # Ptrain, Pval, Ptest = train['samples'].transpose(1, 2), val['samples'].transpose(1, 2), test['samples'].transpose(1, 2)
    # ytrain, yval, ytest = train['labels'], val['labels'], test['labels']

    X, y = torch.load(os.path.join(base_path, 'all_ECG.pt'))

    Ptrain, ytrain = X[idx_train], y[idx_train]
    Pval, yval = X[idx_val], y[idx_val]
    Ptest, ytest = X[idx_test], y[idx_test]

    T, F = Ptrain[0].shape
    D = 1

    Ptrain_static_tensor = np.zeros((len(Ptrain), D))

    mf, stdf = getStats(Ptrain)
    #print('Before tensor_normalize_other', Ptrain.shape)
    Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_ECG(Ptrain, ytrain, mf, stdf)
    Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_ECG(Pval, yval, mf, stdf)
    Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_ECG(Ptest, ytest, mf, stdf)
    #print('After tensor_normalize (X)', Ptrain_tensor.shape)

    Ptrain_tensor = Ptrain_tensor.permute(2, 0, 1)
    Pval_tensor = Pval_tensor.permute(2, 0, 1)
    Ptest_tensor = Ptest_tensor.permute(2, 0, 1)

    #print('Before s-permute', Ptrain_time_tensor.shape)
    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    # print('X', Ptrain_tensor)
    # print('time', Ptrain_time_tensor)
    print('X', Ptrain_tensor.shape)
    print('time', Ptrain_time_tensor.shape)
    # print('time of 0', Ptrain_time_tensor.sum())
    # print('train under 0', (Ptrain_tensor > 1e-10).sum() / Ptrain_tensor.shape[1])
    #print('After s-permute', Ptrain_time_tensor.shape)
    #exit()
    train_chunk = ECGchunk(Ptrain_tensor, None, Ptrain_time_tensor, ytrain_tensor, device = device)
    val_chunk = ECGchunk(Pval_tensor, None, Pval_time_tensor, yval_tensor, device = device)
    test_chunk = ECGchunk(Ptest_tensor, None, Ptest_time_tensor, ytest_tensor, device = device)

    return train_chunk, val_chunk, test_chunk

mitecg_base_path = '/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/datasets/MITECG'
def process_MITECG(split_no = 1, device = None, hard_split = False, normalize = False, exclude_pac_pvc = False, balance_classes = False, div_time = False, 
        need_binarize = False, base_path = mitecg_base_path):

    split_path = 'split={}.pt'.format(split_no)
    idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, split_path))
    if hard_split:
        X = torch.load(os.path.join(base_path, 'all_data/X.pt'))
        y = torch.load(os.path.join(base_path, 'all_data/y.pt')).squeeze()

        # Make times on the fly:
        times = torch.zeros(X.shape[0],X.shape[1])
        for i in range(X.shape[1]):
            times[:,i] = torch.arange(360)

        saliency = torch.load(os.path.join(base_path, 'all_data/saliency.pt'))
        
    else:
        X, times, y = torch.load(os.path.join(base_path, 'all_data.pt'))

    Ptrain, time_train, ytrain = X[:,idx_train,:].float(), times[:,idx_train], y[idx_train].long()
    Pval, time_val, yval = X[:,idx_val,:].float(), times[:,idx_val], y[idx_val].long()
    Ptest, time_test, ytest = X[:,idx_test,:].float(), times[:,idx_test], y[idx_test].long()

    if normalize:

        # Get mean, std of the whole sample from training data, apply to val, test:
        mu = Ptrain.mean()
        std = Ptrain.std()
        Ptrain = (Ptrain - mu) / std
        Pval = (Pval - mu) / std
        Ptest = (Ptest - mu) / std


    if div_time:
        time_train = time_train / 60.0
        time_val = time_val / 60.0
        time_test = time_test / 60.0

    if exclude_pac_pvc:
        train_mask_in = (ytrain < 3)
        Ptrain = Ptrain[:,train_mask_in,:]
        time_train = time_train[:,train_mask_in]
        ytrain = ytrain[train_mask_in]

        val_mask_in = (yval < 3)
        Pval = Pval[:,val_mask_in,:]
        time_val = time_val[:,val_mask_in]
        yval = yval[val_mask_in]

        test_mask_in = (ytest < 3)
        Ptest = Ptest[:,test_mask_in,:]
        time_test = time_test[:,test_mask_in]
        ytest = ytest[test_mask_in]
    
    if need_binarize:
        ytrain = (ytrain > 0).long()
        ytest = (ytest > 0).long()
        yval = (yval > 0).long()

    if balance_classes:
        diff_to_mask = (ytrain == 0).sum() - (ytrain == 1).sum()
        all_zeros = (ytrain == 0).nonzero(as_tuple=True)[0]
        mask_out = all_zeros[:diff_to_mask]
        to_mask_in = torch.tensor([not (i in mask_out) for i in torch.arange(Ptrain.shape[1])])
        print('Num before', (ytrain == 0).sum())
        Ptrain = Ptrain[:,to_mask_in,:]
        time_train = time_train[:,to_mask_in]
        ytrain = ytrain[to_mask_in]
        print('Num after 0', (ytrain == 0).sum())
        print('Num after 1', (ytrain == 1).sum())

        diff_to_mask = (yval == 0).sum() - (yval == 1).sum()
        all_zeros = (yval == 0).nonzero(as_tuple=True)[0]
        mask_out = all_zeros[:diff_to_mask]
        to_mask_in = torch.tensor([not (i in mask_out) for i in torch.arange(Pval.shape[1])])
        print('Num before', (yval == 0).sum())
        Pval = Pval[:,to_mask_in,:]
        time_val = time_val[:,to_mask_in]
        yval = yval[to_mask_in]
        print('Num after 0', (yval == 0).sum())
        print('Num after 1', (yval == 1).sum())

        diff_to_mask = (ytest == 0).sum() - (ytest == 1).sum()
        all_zeros = (ytest == 0).nonzero(as_tuple=True)[0]
        mask_out = all_zeros[:diff_to_mask]
        to_mask_in = torch.tensor([not (i in mask_out) for i in torch.arange(Ptest.shape[1])])
        print('Num before', (ytest == 0).sum())
        Ptest = Ptest[:,to_mask_in,:]
        time_test = time_test[:,to_mask_in]
        ytest = ytest[to_mask_in]
        print('Num after 0', (ytest == 0).sum())
        print('Num after 1', (ytest == 1).sum())

    train_chunk = ECGchunk(Ptrain, None, time_train, ytrain, device = device)
    val_chunk = ECGchunk(Pval, None, time_val, yval, device = device)
    test_chunk = ECGchunk(Ptest, None, time_test, ytest, device = device)

    print('Num after 0', (yval == 0).sum())
    print('Num after 1', (yval == 1).sum())
    print('Num after 0', (ytest == 0).sum())
    print('Num after 1', (ytest == 1).sum())

    if hard_split:
        gt_exps = saliency.transpose(0,1).unsqueeze(-1)[:,idx_test,:]
        if exclude_pac_pvc:
            gt_exps = gt_exps[:,test_mask_in,:]
        return train_chunk, val_chunk, test_chunk, gt_exps
    else:
        return train_chunk, val_chunk, test_chunk
    

def process_MITECG_for_WINIT(
                             device = None, 
                             hard_split = False, 
                             normalize = False, 
                             exclude_pac_pvc = False, 
                             balance_classes = False, 
                             div_time = False, 
                             need_binarize = False, 
                             base_path = mitecg_base_path
                             ):

    # split_path = 'split={}.pt'.format(split_no)
    # idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, split_path))
    if hard_split:
        X = torch.load(os.path.join(base_path, 'all_data/X.pt'))
        y = torch.load(os.path.join(base_path, 'all_data/y.pt')).squeeze()

        # Make times on the fly:
        times = torch.zeros(X.shape[0],X.shape[1])
        for i in range(X.shape[1]):
            times[:,i] = torch.arange(360)

        saliency = torch.load(os.path.join(base_path, 'all_data/saliency.pt'))
        
    else:
        X, times, y = torch.load(os.path.join(base_path, 'all_data.pt'))

    P_all, time_all, y_all = X.float(), times, y.long()



    if div_time:
        time_all = time_all / 60.0

    if exclude_pac_pvc:
        all_mask_in = (y_all < 3)
        P_all = P_all[:,all_mask_in,:]
        time_all = time_all[:,all_mask_in]
        y_all = y_all[all_mask_in]
    
    if need_binarize:
        y_all = (y_all > 0).long()

    if balance_classes:
        diff_to_mask = (y_all == 0).sum() - (y_all == 1).sum()
        all_zeros = (y_all == 0).nonzero(as_tuple=True)[0]
        mask_out = all_zeros[:diff_to_mask]
        to_mask_in = torch.tensor([not (i in mask_out) for i in torch.arange(P_all.shape[1])])
        print('Num before', (y_all == 0).sum())
        P_all = P_all[:,to_mask_in,:]
        time_all = time_all[:,to_mask_in]
        y_all = y_all[to_mask_in]
        print('Num after 0', (y_all == 0).sum())
        print('Num after 1', (y_all == 1).sum())

    all_chunk = ECGchunk(P_all, None, time_all, y_all, device=device)

    print('Num after 0', (y_all == 0).sum())
    print('Num after 1', (y_all == 1).sum())    
    
    if hard_split: # TODO only if gt_exps is required
        gt_exps = saliency.transpose(0,1).unsqueeze(-1)[:,:,:]
        if exclude_pac_pvc:
            gt_exps = gt_exps[:,all_mask_in,:]

        return all_chunk, gt_exps
    else:
        return all_chunk

boiler_base_path = "data/Boiler"
def process_Boiler_WinIT(device = None, base_path = boiler_base_path, normalize = False):
    # x_full = torch.load(os.path.join(base_path, 'xfull.pt')).to(device).float()
    # y_full = torch.load(os.path.join(base_path, 'yfull.pt')).to(device).long()
    # sfull = torch.load(os.path.join(base_path, 'sfull.pt')).to(device).float()
    train, val, test = torch.load(os.path.join(base_path, 'split=1.pt'))#.to(device).float()
    # print('s', sfull.shape)
    # print('xfull', x_full.shape)
    # print('yfull', y_full.shape)
    print('all len', len(all))
    print('all[0].shape', all[0].shape)
    print('all[1].shape', all[1].shape)
    print('all[2].shape', all[2].shape)
    # exit()

    T_full = torch.zeros(36, x_full.shape[1]).to(device)
    for i in range(T_full.shape[1]):
        T_full[:,i] = torch.arange(36)

    idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, 'split={}.pt'.format(split_no)))

    train_d = [x_full[:,:,:], T_full[:,:], y_full[:]]
    val_d = [x_full[:,:,:], T_full[:,:], y_full[:]]
    test_d = [x_full[:,:,:], T_full[:,:], y_full[:]]

    stest = sfull[:,:,:]

    return train_d, val_d, test_d, stest


class EpiDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y, augment_negative = None):
        self.X = X # Shape: (T, N, d)
        self.times = times # Shape: (T, N)
        self.y = y # Shape: (N,)

        #self.augment_negative = augment_negative
        if augment_negative is not None:
            mu, std = X.mean(dim=1), X.std(dim=1, unbiased = True)
            num = int(self.X.shape[1] * augment_negative)
            Xnull = torch.stack([mu + torch.randn_like(std) * std for _ in range(num)], dim=1).to(self.X.get_device())

            self.X = torch.cat([self.X, Xnull], dim=1)
            extra_times = torch.arange(self.X.shape[0]).to(self.X.get_device())
            self.times = torch.cat([self.times, extra_times.unsqueeze(1).repeat(1, num)], dim = -1)
            self.y = torch.cat([self.y, (torch.ones(num).to(self.X.get_device()).long() * 2)], dim = 0)

        # print('X', self.X.shape)
        # print('times', self.times.shape)
        # print('y', self.y.shape)
        # exit()

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y 

epi_base_path = '/home/owq978/TimeSeriesXAI/ECGdata/Epilepsy'
def process_Epilepsy(split_no = 1, device = None, base_path = epi_base_path):

    split_path = 'split_{}.npy'.format(split_no)
    idx_train, idx_val, idx_test = np.load(os.path.join(base_path, split_path), allow_pickle = True)

    # Ptrain, Pval, Ptest = train['samples'].transpose(1, 2), val['samples'].transpose(1, 2), test['samples'].transpose(1, 2)
    # ytrain, yval, ytest = train['labels'], val['labels'], test['labels']

    X, y = torch.load(os.path.join(base_path, 'all_epilepsy.pt'))

    Ptrain, ytrain = X[idx_train], y[idx_train]
    Pval, yval = X[idx_val], y[idx_val]
    Ptest, ytest = X[idx_test], y[idx_test]

    T, F = Ptrain[0].shape
    D = 1

    Ptrain_static_tensor = np.zeros((len(Ptrain), D))

    mf, stdf = getStats(Ptrain)
    #print('Before tensor_normalize_other', Ptrain.shape)
    Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_ECG(Ptrain, ytrain, mf, stdf)
    Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_ECG(Pval, yval, mf, stdf)
    Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_ECG(Ptest, ytest, mf, stdf)
    #print('After tensor_normalize (X)', Ptrain_tensor.shape)

    Ptrain_tensor = Ptrain_tensor.permute(2, 0, 1)
    Pval_tensor = Pval_tensor.permute(2, 0, 1)
    Ptest_tensor = Ptest_tensor.permute(2, 0, 1)

    #print('Before s-permute', Ptrain_time_tensor.shape)
    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    # print('X', Ptrain_tensor)
    # print('time', Ptrain_time_tensor)
    print('X', Ptrain_tensor.shape)
    print('time', Ptrain_time_tensor.shape)

    train_chunk = ECGchunk(Ptrain_tensor, None, Ptrain_time_tensor, ytrain_tensor, device = device)
    val_chunk = ECGchunk(Pval_tensor, None, Pval_time_tensor, yval_tensor, device = device)
    test_chunk = ECGchunk(Ptest_tensor, None, Ptest_time_tensor, ytest_tensor, device = device)

    return train_chunk, val_chunk, test_chunk

def decomposition_statistics(pool_layer, X):

    # Decomposition by trend layer:
    trend = pool_layer(X)
    seasonal = X - trend

    d = {
        'mu_trend': trend.mean(dim=1),
        'std_trend': trend.std(unbiased = True, dim = 1),
        'mu_seasonal': seasonal.mean(dim=1),
        'std_seasonal': seasonal.std(unbiased = True, dim = 1)
    }

    return d


def process_Boiler(split_no = 1, device = None, base_path = boiler_base_path, normalize = False):
    x_full = torch.load(os.path.join(base_path, 'xfull.pt')).to(device).float()
    y_full = torch.load(os.path.join(base_path, 'yfull.pt')).to(device).long()
    sfull = torch.load(os.path.join(base_path, 'sfull.pt')).to(device).float()
    print('s', sfull.shape)
    print('xfull', x_full.shape)
    print('yfull', y_full.shape)
    # exit()

    T_full = torch.zeros(36, x_full.shape[1]).to(device)
    for i in range(T_full.shape[1]):
        T_full[:,i] = torch.arange(36)

    idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, 'split={}.pt'.format(split_no)))

    train_d = [x_full[:,idx_train,:], T_full[:,idx_train], y_full[idx_train]]
    val_d = [x_full[:,idx_val,:], T_full[:,idx_val], y_full[idx_val]]
    test_d = [x_full[:,idx_test,:], T_full[:,idx_test], y_full[idx_test]]

    stest = sfull[:,idx_test,:]

    return train_d, val_d, test_d, stest


def process_Boiler_OLD(split_no = 1, train_ratio=0.8, device = None, base_path = boiler_base_path):
    data = pd.read_csv(os.path.join(base_path, 'full.csv')).values
    data = data[:, 2:]  #remove time step

    window_size = 6
    segments_length = [1, 2, 3, 4, 5, 6]

    # Load path

    print('positive sample size:',sum(data[:,-1]))
    feature, label = [], []
    for i in range(window_size - 1, len(data)):
        label.append(data[i, -1])

        sample = []
        for length in segments_length:
            a = data[(i- length + 1):(i + 1), :-1]
            a = np.pad(a,pad_width=((0,window_size -length),(0,0)),mode='constant')# padding to [window_size, x_dim]
            sample.append(a)

        sample = np.array(sample)
        sample = np.transpose(sample,axes=((2,0,1)))[:,:,:]

        feature.append(sample)

    feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.int32)

    x_full = torch.tensor(feature.reshape(*feature.shape[:-2], -1)).permute(2,0,1)
    y_full = torch.from_numpy(label)

    # Make times:
    T_full = torch.zeros(36, x_full.shape[1])
    for i in range(T_full.shape[1]):
        T_full[:,i] = torch.arange(36)

    # Now split:
    # idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, 'split={}.pt'.format(split_no)))

    # x_full, T_full, y_full = x_full.to(device), T_full.to(device), y_full.to(device).long()

    # train_d = (x_full[:,idx_train,:], T_full[:,idx_train], y_full[idx_train])
    # val_d = (x_full[:,idx_val,:], T_full[:,idx_val], y_full[idx_val])
    # test_d = (x_full[:,idx_test,:], T_full[:,idx_test], y_full[idx_test])



    x_cpu = x_full.cpu()
    T_cpu = T_full.cpu()
    y_cpu = y_full.cpu()

    # 2) total number of samples
    N = x_cpu.shape[1]
    all_idx = np.arange(N)

    # 3) split off the training set
    idx_train, idx_temp = train_test_split(
        all_idx,
        train_size=train_ratio,
        random_state=42,
        shuffle=True
    )

    # 4) split the remainder equally into val/test
    if len(idx_temp) > 0:
        idx_val, idx_test = train_test_split(
            idx_temp,
            train_size=0.5,
            random_state=42,
            shuffle=True
        )
    else:
        # edge case: no leftover samples
        idx_val, idx_test = np.array([], dtype=int), np.array([], dtype=int)

    # 5) build the three tuples exactly as before
    #    note x_full shape is (T, N, F), so we index on dim=1
    train_d = (
        x_cpu[:, idx_train, :].to(device),
        T_cpu[:, idx_train].to(device),
        y_cpu[idx_train].to(device),
    )
    val_d = (
        x_cpu[:, idx_val, :].to(device),
        T_cpu[:, idx_val].to(device),
        y_cpu[idx_val].to(device),
    )
    test_d = (
        x_cpu[:, idx_test, :].to(device),
        T_cpu[:, idx_test].to(device),
        y_cpu[idx_test].to(device),
    )

    for split_name, split_d in [("Train", train_d), ("Val", val_d), ("Test", test_d)]:
        y_split = split_d[2]                    # tensor of shape (n_samples,)
        y_np    = y_split.detach().cpu().numpy() 
        classes, counts = np.unique(y_np, return_counts=True)
        dist = dict(zip(classes.tolist(), counts.tolist()))
        print(f"{split_name} class distribution: {dist}")
        
    return train_d, val_d, test_d


spike_path = '/home/owq978/TimeSeriesXAI/datasets/Spike/'
def process_Synth(split_no = 1, device = None, base_path = spike_path, regression = False,
        label_noise = None):

    split_path = os.path.join(base_path, 'split={}.pt'.format(split_no))
    print("split_path：", split_path)

    D = torch.load(split_path,  weights_only=False)

    D['train_loader'].X = D['train_loader'].X.float().to(device)
    D['train_loader'].times = D['train_loader'].times.float().to(device)
    if regression:
        D['train_loader'].y = D['train_loader'].y.float().to(device)
    else:
        D['train_loader'].y = D['train_loader'].y.long().to(device)

    val = []
    val.append(D['val'][0].float().to(device))
    val.append(D['val'][1].float().to(device))
    val.append(D['val'][2].long().to(device))
    if regression:
        val[-1] = val[-1].float()
    D['val'] = tuple(val)

    test = []
    test.append(D['test'][0].float().to(device))
    test.append(D['test'][1].float().to(device))
    test.append(D['test'][2].long().to(device))
    if regression:
        test[-1] = test[-1].float()
    D['test'] = tuple(test)

    if label_noise is not None:
        # Find some samples in training to switch labels:

        to_flip = int(label_noise * D['train_loader'].y.shape[0])
        to_flip = to_flip + 1 if (to_flip % 2 == 1) else to_flip # Add one if it isn't even

        flips = torch.randperm(D['train_loader'].y.shape[0])[:to_flip]

        max_label = D['train_loader'].y.max()

        for i in flips:
            D['train_loader'].y[i] = (D['train_loader'].y[i] + 1) % max_label

    return D