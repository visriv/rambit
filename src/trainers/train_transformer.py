import sys, os
import torch
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, precision_recall_fscore_support, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_absolute_error
sys.path.append(os.path.dirname(__file__))

from src.utils.predictors.loss import Poly1CrossEntropyLoss
from src.models.run_model_utils import batch_forwards_TransformerMVTS
from src.models.encoders.simple import CNN, LSTM

default_scheduler_args = {
    'mode': 'max', 
    'factor': 0.1, 
    'patience': 10,
    'threshold': 0.00001, 
    'threshold_mode': 'rel',
    'cooldown': 0, 
    'min_lr': 1e-8, 
    'eps': 1e-08, 
    'verbose': True
}

def one_hot(y_):
    # Convert y_ to one-hot
    if not (type(y_) is np.ndarray):
        y_ = y_.detach().clone().cpu().numpy() # Assume it's a tensor
    
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def train(
        model, 
        train_loader, 
        val_tuple, # Must be (X, times, y) all in one tensor for each
        n_classes, 
        num_epochs,
        class_weights = None, 
        optimizer = None, 
        standardize = False,
        save_path = None,
        validate_by_step = None,
        criterion = None,
        scheduler_args = default_scheduler_args,
        show_sizes = False,
        regression = False,
        use_scheduler = True,
        counterfactual_training = False,
        max_mask_size = None,
        replace_method = None,
        print_freq = 10,
        clip_grad = None,
        detect_irreg = False,
        ):
    '''
    Loader should output (B, d, T) - in style of captum input

    Params:
        rand_mask_size (default: None): If an integer is provided, trains model with a 
            random mask generated at test time
        counterfactual_training (bool, optional): If True, counterfactually trains
            the model, as in Hase et al., 2021
        max_mask_size (int, optional): Maximum mask size for counterfactual training 
            procedure
        replace_method (callable, optional): Replacement method to replace values in
            the input when masked out
    '''
    
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01)

    if criterion is None: # Set if not given
        if regression:
            criterion = torch.nn.MSELoss()
        else:
            criterion = Poly1CrossEntropyLoss(
                num_classes = n_classes,
                epsilon = 1.0,
                weight = class_weights,
                reduction = 'mean'
            )

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

    if save_path is None:
        save_path = 'tmp.pt'

    train_loss, val_auc = [], []
    max_val_auc, best_epoch = 0, 0
    for epoch in range(num_epochs):
        
        # Train:
        model.train()
        for X, times, y in train_loader:

            #print(X.detach().clone().cpu().numpy())
            #print(times.detach().clone().cpu().numpy())
            
            # if detect_irreg:
            #     src_mask = (times == 0)
            #     out = model(X, times, captum_input = True, show_sizes = show_sizes, src_mask = src_mask)
            out = model(X, times, captum_input = True, show_sizes = show_sizes)

            optimizer.zero_grad()
            loss = criterion(out, y)
            loss.backward()

            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            if counterfactual_training:
                # calculate loss on replaced values and update
                # Sample 1/2 of batch, run replace:
                batch_size, T, d = X.shape[0], X.shape[1], X.shape[2]
                x_inds = torch.randperm(batch_size)[:(batch_size // 2)]
                xsamp = X[x_inds,:,:]
                masks = torch.ones_like(xsamp).float().to(xsamp.device)
                
                # determine max mask size to sample out (mms)
                mms = max_mask_size if max_mask_size is not None else T * d
                if mms < 1 and mms > 0:
                    mms = X.shape[0] * X.shape[1] * mms # If proportion is provided
                mask_nums = torch.randint(0, high = int(mms), size = ((batch_size // 2),)) 

                # Fill in masks:
                for i in range(masks.shape[0]):
                    cart = torch.cartesian_prod(torch.arange(T), torch.arange(d))[:mask_nums[i]]
                    masks[i,cart[:,0],cart[:,1]] = 0 # Set all spots to 1
                xmasked = replace_method(xsamp, masks)

                out = model(xmasked, times[x_inds,:], captum_input = True, show_sizes = show_sizes)

                optimizer.zero_grad()
                loss2 = criterion(out, y[x_inds])
                loss2.backward()
                optimizer.step()

                loss = loss + loss2 # Add together total loss to be shown in train_loss

            train_loss.append(loss.item())
                    
        # Validation:
        model.eval()
        with torch.no_grad():
            X, times, y = val_tuple
            if validate_by_step is not None:
                if isinstance(model, CNN) or isinstance(model, LSTM):
                    pred = torch.cat(
                        [model(xb, tb) for xb, tb in zip(torch.split(X, validate_by_step, dim=1),
                                                        torch.split(times, validate_by_step, dim=1))],
                        dim=0
                    )
                else:
                    pred, _ = batch_forwards_TransformerMVTS(model, X, times, batch_size=validate_by_step)
            else:
                pred = model(X, times, show_sizes=show_sizes)

            val_loss = criterion(pred, y)

            if regression:
                # Keep your existing behavior
                auc = -1.0 * mean_absolute_error(y.cpu().numpy(), pred.cpu().numpy())
                # No PR/AUROC for regression; set to NaN for printing consistency
                auroc = np.nan
                auprc = np.nan
                precision = np.nan
                recall = np.nan
                f1 = np.nan
            else:
                # ---- Classification metrics ----
                y_true = y.cpu().numpy()
                # Use softmax probabilities for AUROC/AUPRC
                probs = torch.softmax(pred, dim=1).detach().cpu().numpy()
                y_pred = probs.argmax(axis=1)

                # Macro-averaged AUROC (multiclass-safe)
                auroc = np.nan
                auprc = np.nan
                try:
                    if np.unique(y_true).size >= 2:
                        auroc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
                        # AUPRC via label binarization (macro average)
                        y_bin = label_binarize(y_true, classes=np.arange(probs.shape[1]))
                        auprc = average_precision_score(y_bin, probs, average="macro")
                except ValueError:
                    auroc = np.nan
                    auprc = np.nan

                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0
                )

                # Preserve your original scheduler signal: use F1 as "auc"
                auc = f1

            if use_scheduler:
                scheduler.step(auc)  # Step the scheduler

            val_auc.append(auc)

            if auc > max_val_auc:
                max_val_auc = auc
                best_epoch = epoch
                best_auprc = auprc
                best_precision = precision
                best_recall = recall
                best_f1 = f1
                torch.save(model.state_dict(), save_path)

        if (epoch + 1) % print_freq == 0:  # Print progress:
            met = 'MAE' if regression else 'F1'
            # For convenience, print both "AUC" and "AUROC" (same value here)
            print(
                'Epoch {}, Train Loss = {:.4f}, Val Loss = {:.4f}, {} (scheduler) = {:.4f} | '
                'AUROC = {:.4f}, AUC = {:.4f}, AUPRC = {:.4f}, Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f}'.format(
                    epoch + 1,
                    train_loss[-1],
                    float(val_loss.detach().cpu().item()),
                    met, float(auc),
                    float(auroc) if not np.isnan(auroc) else np.nan,
                    float(auroc) if not np.isnan(auroc) else np.nan,  # AUC alias for AUROC
                    float(auprc) if not np.isnan(auprc) else np.nan,
                    float(precision) if not np.isnan(precision) else np.nan,
                    float(recall) if not np.isnan(recall) else np.nan,
                    float(f1) if not np.isnan(f1) else np.nan
                )
            )


    # Return best model:
    model.load_state_dict(torch.load(save_path))

    if save_path == 'tmp.pt':
        os.remove('tmp.pt') # Remove temporarily stored file

    # print('Epoch = {}, AUC = {:.4f}'.format(best_epoch, max_val_auc))


    print(
        'Best AUC achieved at Epoch {} | '
        'AUROC = {:.4f},  AUPRC = {:.4f}, Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f}'.format(
            best_epoch,
            float(max_val_auc) if not np.isnan(max_val_auc) else np.nan,
            float(best_auprc) if not np.isnan(best_auprc) else np.nan,
            float(best_precision) if not np.isnan(best_precision) else np.nan,
            float(best_recall) if not np.isnan(best_recall) else np.nan,
            float(best_f1) if not np.isnan(best_f1) else np.nan
        )
    )
    

    return model, train_loss, val_auc