import torch
from torch.distributions import Bernoulli
from typing import Tuple, Optional
from src.explainer.explainers import BaseExplainer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from itertools import product

class JIMEx(BaseExplainer):
    def __init__(
        self,
        num_features: int,
        num_samples: int = 3,
        num_masks: int = 3,
        wt_bounds: Tuple[int, int] = (1, 3),
        wd_bounds: Tuple[int, int] = (1, 3),
        device: Optional[torch.device] = None,
        metric: str = "pd",
        window_size: int = 10,
        train_loader: DataLoader  = None,
        random_state: int  = None,
        all_zero_cf: bool = False
    ):
        """
        Args:
          num_masks:       L, number of random masks
          wt_bounds:       (Wt_min, Wt_max) time‐segment length bounds
          wd_bounds:       (Wd_min, Wd_max) feature‐segment width bounds
          device:          torch device (cpu / cuda); inferred from model if None
          metric: str = "pd",
        """
        # self.model = model.eval()
        self.L     = num_masks
        self.Wt_min, self.Wt_max = wt_bounds
        self.Wd_min, self.Wd_max = wd_bounds
        self.device = device or next(self.model.parameters()).device
        self.metric = metric
        self.window_size = window_size
        self.num_samples = num_samples
        self.num_features = num_features


        if train_loader is not None:
            self.data_distribution = (
                torch.stack([x[0] for x in train_loader.dataset]).detach().cpu().numpy()
            )
        else:
            self.data_distribution = None
        self.rng = np.random.default_rng(random_state)
        self.all_zero_cf = all_zero_cf
    def _model_predict(self, x):
        """
        Run predict on base model. If the output is binary, i.e. num_class = 1, we will make it
        into a probability distribution by append (p, 1-p) to it.
        """
        p = self.base_model.predict(x, return_all=False)
        if self.base_model.num_states == 1:
            # Create a 'probability distribution' (p, 1 - p)
            prob_distribution = torch.cat((p, 1 - p), dim=1)
            return prob_distribution
        return p
    
    
    def attribute(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        B, D, T = X.shape

        I_all = torch.zeros((B, T, D), device=self.device, dtype=X.dtype)
        for t, d in tqdm(product(range(T), range(D)), total=T*D):
            I_all[:, t, d] = self.attribute_one_cell(X, t, d, self.all_zero_cf)
            # print("in JImex, one cell attributed, the 5 sample values of this cell are:", (t,d), I_all[:5, t, d] )
        return I_all.permute(0,2,1).detach().cpu().numpy() # return in B x D x T shape



    def attribute_one_cell(
        self,
        X: torch.Tensor,
        source_t: int,
        source_d: int,
        all_zero_cf: bool
    ) -> torch.Tensor:
        """
        Compute JIMEx attribution for cell (source_t, source_d) over a batch.
        Returns a (B,) tensor with one score per item in the batch.
        """

        X = X.to(self.device)
        B, D, T = X.shape



        # accumulate Δ per-source cell, per sample B
        I_cell = torch.zeros(B, device=self.device, dtype=X.dtype)

        all_M1 = []
        all_M2 = []
        t_max = []

        # generate all the masks and save them in a list
        for _ in range(self.L):
            # window sizes bounded by data dims
            max_Wt = min(self.Wt_max, T)
            min_Wt = min(self.Wt_min, max_Wt)
            Wt = torch.randint(min_Wt, max_Wt + 1, (), device=self.device).item()

            max_Wd = min(self.Wd_max, D)
            min_Wd = min(self.Wd_min, max_Wd)
            Wd = torch.randint(min_Wd, max_Wd + 1, (), device=self.device).item()

            # valid start positions
            t0 = torch.randint(0, T - Wt + 1, (), device=self.device).item()
            d0 = torch.randint(0, D - Wd + 1, (), device=self.device).item()

            # base mask M: (T, D)
            M = torch.ones((T, D), device=self.device, dtype=X.dtype)
            M[t0:t0 + Wt, d0:d0 + Wd] = 0.0

            # ensure source is kept in M2
            M[source_t, source_d] = 1.0

            # reshape to match X (B, D, T)
            M = M.permute(1, 0).unsqueeze(0)  # (1, D, T)

            # M2 keeps target bit; M1 flips it so M1/M2 differ at (t*,d*)
            M2 = M.clone()
            M1 = M.clone()
            M1[:, source_d, source_t] = 0.0

           

            all_M1.append(M1)
            all_M2.append(M2)
            t_max.append(t0 + Wt)


        # compute and use the cached CF tensor
        CF = self._compute_cf(B, T-1, X, all_zero_cf) # output: ( num_samples, B, D T) 
        num_samples = CF.shape[0]

        # Now iterate over masks and obtain importance score based on prediction at t_target = t_max + w
        for l in range(self.L):
            I_cell_this_mask = 0
            count_valid_windows = 0


            t_first = t_max[l]
            t_last = min(T - 1, t_max[l] + self.window_size - 1)

            for w in range(1, self.window_size + 1):
                t_target = t_max[l] + w - 1
                if (t_target >= T):
                    continue
                

                count_valid_windows += 1
                CF_sbdt = CF.to(X.dtype)  # (S, B, D, T)

                # Broadcast X and masks to (S, B, D, T) WITHOUT merging S and B
                # X: (B, D, T) -> (S, B, D, T)
                X_sbdt = X.unsqueeze(0).expand(num_samples, -1, -1, -1)

                # all_M1[l] / all_M2[l] are (1, D, T); make them (num_samples, B, D, T) by broadcast
                M1_sbdt = all_M1[l].unsqueeze(0)#.unsqueeze(1)  # (1, 1, D, T), will bcast to (num_samples, B, D, T)
                M2_sbdt = all_M2[l].unsqueeze(0)#.unsqueeze(1)

                # print("num_samples", num_samples )
                # print("X_sbdt.shape:", X_sbdt.shape )
                # print("M1_sbdt.shape:", M1_sbdt.shape )
                # print("M2_sbdt.shape:", M2_sbdt.shape )
                # Construct masked inputs at full length T; we'll slice in time below
                X1_sbdt = X_sbdt * M1_sbdt + (1.0 - M1_sbdt) * CF_sbdt  # (num_samples, B, D, T)
                X2_sbdt = X_sbdt * M2_sbdt + (1.0 - M2_sbdt) * CF_sbdt  # (num_samples, B, D, T)

                # ---- model predictions ----
                with torch.no_grad():
                    # f_orig is the same across samples (no CF), shape (B,)
                    f_orig = self._model_predict(X[:, :, 0:t_target + 1])  # (B,)

                    # Evaluate per-sample so we don't merge S and B
                    f_masked1_list = []
                    f_masked2_list = []
                    for s in range(num_samples):
                        f_masked1_list.append(
                            self._model_predict(X1_sbdt[s, :, :, 0:t_target + 1])  # (B,)
                        )
                        f_masked2_list.append(
                            self._model_predict(X2_sbdt[s, :, :, 0:t_target + 1])  # (B,)
                        )

                    # Stack to (S, B)
                    f_masked1 = torch.stack(f_masked1_list, dim=0)
                    f_masked2 = torch.stack(f_masked2_list, dim=0)

                # ---- metrics per sample ----
                # compute_metric expects two (B,) tensors; map it over num_samples without merging dims
                i1_list = []
                i2_list = []
                for s in range(num_samples):
                    i1_list.append(self._compute_metric(f_orig, f_masked1[s]))  # (B,)
                    i2_list.append(self._compute_metric(f_orig, f_masked2[s]))  # (B,)

                i1 = torch.stack(i1_list, dim=0)  # (num_samples, B)
                i2 = torch.stack(i2_list, dim=0)  # (num_samples, B)

                # Δ per sample, then mean over the num_samples dim num_samples -> (B,)
                delta_mean = (i1 - i2).mean(dim=0)  # (B,)

                I_cell_this_mask += delta_mean


            if (count_valid_windows == 0):
                I_cell += 0.0
              
            else:
                I_cell += ( I_cell_this_mask / float(count_valid_windows))

        # average over masks → (B,)
        I_cell = I_cell / float(self.L)

       

        return I_cell

    def _compute_metric(self, p_y_exp: torch.Tensor, p_y_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric for comparisons of two distributions.

        Args:
            p_y_exp:
                The current expected distribution. Shape = (batch_size, num_states)
            p_y_hat:
                The modified (counterfactual) distribution. Shape = (batch_size, num_states)

        Returns:
            The result Tensor of shape (batch_size).

        """
        if self.metric == "kl":
            return torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(p_y_hat), p_y_exp), -1)
        if self.metric == "js":
            average = (p_y_hat + p_y_exp) / 2
            lhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_hat)
            rhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_exp)
            return torch.sum((lhs + rhs) / 2, -1)
        if self.metric == "pd":
            diff = torch.abs(p_y_hat - p_y_exp)
            return torch.sum(diff, -1)
        raise Exception(f"unknown metric. {self.metric}")
    

    def _compute_cf(self, B, t, sample_x, all_zero_cf):
        # Compute and cache a CF matrix to replace in the mask later
        ## Sample counterfactuals for every (f, s, b, t)
        batch_size = B
        CF = torch.empty((self.num_samples, batch_size, self.num_features, t+1), device=self.device, dtype=sample_x.dtype)
        # N_hist = self.data_distribution.shape[0] * self.data_distribution.shape[2]  # N_samples × T
        if (all_zero_cf):
            return CF
        

        for f in range(self.num_features):
            # flatten historical values for feature f
            vals = self.data_distribution[:, f, :].reshape(-1)  # shape (N_hist,)
            # draw S×B×T values
            draws = self.rng.choice(vals, size=(self.num_samples, batch_size, t+1))
            CF[:, :, f, :] = torch.from_numpy(draws).to(self.device)

        ## CF generation complete
        return CF
    

    def get_name(self):
        builder = ["jimex", "num_masks", str(self.L), 
                   "window_size", str(self.window_size),
                   "num_samples", str(self.num_samples),
                   "wt_bounds", "_".join([str(self.Wt_min), str(self.Wt_max)])
                   ]

        builder.append(self.metric)
        if self.data_distribution is not None:
            builder.append("usedatadist")
        return "_".join(builder)