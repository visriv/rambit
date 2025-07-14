from __future__ import annotations

import logging
import pathlib
from time import time
import time as time_og
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Optional
from src.explainer.explainers import BaseExplainer
from src.explainer.generator.generator import (
    FeatureGenerator,
    BaseFeatureGenerator,
    GeneratorTrainingResults,
)
from src.explainer.generator.jointgenerator import JointFeatureGenerator


class BiWinITExplainer(BaseExplainer):
    """
    The explainer for our method WinIT
    """

    def __init__(
        self,
        device,
        num_features: int,
        data_name: str,
        path: pathlib.Path,
        train_loader: DataLoader | None = None,
        window_size: int = 10,
        num_samples: int = 3,
        conditional: bool = False,
        joint: bool = False,
        metric: str = "pd",
        # height: int = 2,
        random_state: int | None = None,
        **kwargs,
    ):
        """
        Construtor

        Args:
            device:
                The torch device.
            num_features:
                The number of features.
            data_name:
                The name of the data.
            path:
                The path indicating where the generator to be saved.
            train_loader:
                The train loader if we are using the data distribution instead of a generator
                for generating counterfactual. Default=None.
            window_size:
                The window size for the WinIT
            num_samples:
                The number of Monte-Carlo samples for generating counterfactuals.
            conditional:
                Indicate whether the individual feature generator we used are conditioned on
                the current features. Default=False
            joint:
                Indicate whether we are using the joint generator.
            metric:
                The metric for the measures of comparison of the two distributions for i(S)_a^b
            random_state:
                The random state.
            **kwargs:

        """
        super().__init__(device)
        self.window_size = window_size
        self.num_samples = num_samples
        self.num_features = num_features
        self.data_name = data_name
        self.joint = joint
        self.conditional = conditional
        self.metric = metric
        self.mask_strategy = kwargs['mask_strategy'],
        self.height = kwargs['height']
        self.generators: BaseFeatureGenerator | None = None
        self.path = path
        if train_loader is not None:
            self.data_distribution = (
                torch.stack([x[0] for x in train_loader.dataset]).detach().cpu().numpy()
            )
        else:
            self.data_distribution = None
        self.rng = np.random.default_rng(random_state)
        self.k_upper_triangular_mask = self._precompute_k()
        self.log = logging.getLogger(BiWinITExplainer.__name__)
        if len(kwargs):
            self.log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

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

    def _precompute_k(self):
        # k[n, t] = how far down from d_start we go at offset n, time t
        W = self.window_size
        T = self.max_timesteps  # or pass in when you know it
        k = np.zeros((W, T), dtype=int)
        for n in range(1, W+1):
            span = n - 1
            for t in range(T):
                t_s = t - span
                if span > 0 and t_s >= 0:
                    α = (t - t_s) / span
                    k[n-1, t] = int(round((α**self.slope)*(self.height - 1)))
                else:
                    k[n-1, t] = 0
        return k
    
    def attribute(self, x):
        """
        Compute the WinIT attribution.

        Args:
            x:
                The input Tensor of shape (batch_size, num_features, num_times)

        Returns:
            The attribution Tensor of shape (batch_size, num_features, num_times, window_size)
            The (i, j, k, l)-entry is the importance of observation (i, j, k - window_size + l + 1)
            to the prediction at time k

        """
        self.base_model.eval()
        self.base_model.zero_grad()

        with torch.no_grad():
            tic = time()

            batch_size, num_features, num_timesteps = x.shape
            scores = []
            i_ab_S1_array = np.zeros((num_timesteps, num_features, self.window_size, batch_size), dtype=float)
            i_ab_S2_array = np.zeros((num_timesteps, num_features, self.window_size, batch_size), dtype=float)
            IS_array = np.zeros((num_timesteps, num_features, self.window_size, batch_size), dtype=float)
            for t in tqdm(range(num_timesteps), desc="BiWinIT Time-step loop", total=num_timesteps):

                window_size = min(t, self.window_size)

                if t == 0:
                    scores.append(np.zeros((batch_size, num_features, self.window_size)))
                    continue

                # x = (num_sample, num_feature, n_timesteps)
                p_y = self._model_predict(x[:, :, : t + 1])
                        # Compute P = p(y_t | X_{1:t})
                p_y_exp = (
                    p_y.unsqueeze(0)
                    .repeat(self.num_samples, 1, 1)
                    .reshape(self.num_samples * batch_size, p_y.shape[-1])
                )

                # Compute and cache a CF matrix to replace in the mask later
                ## Sample counterfactuals for every (f, s, b, t)
                CF = torch.empty((self.num_samples, batch_size, self.num_features, t+1), device=self.device, dtype=x.dtype)
                # N_hist = self.data_distribution.shape[0] * self.data_distribution.shape[2]  # N_samples × T

                for f in range(self.num_features):
                    # flatten historical values for feature f
                    vals = self.data_distribution[:, f, :].reshape(-1)  # shape (N_hist,)
                    # draw S×B×T values
                    draws = self.rng.choice(vals, size=(self.num_samples, batch_size, t+1))
                    CF[:, :, f, :] = torch.from_numpy(draws).to(self.device)

                ## CF generation complete

                for n in range(window_size):
                    time_past = t - n
                    delta_time = n + 1

                    # counterfactual shape = (num_feat, num_samples, batch_size, time_forward)
                    for f in range(num_features):
                        ## Set S1 ##
                        #out shape should be self.num_features, self.num_samples, batch_size, t+1
                        x_rep_in_1, x_rep_in_2 = self.replace_cfs(x[:, :, : t + 1], 
                                               t_start = t-n-1,
                                               t_end = t,
                                               CF = CF,
                                               mask_strategy = self.mask_strategy,
                                               d_start = f,
                                               height = self.height,
                                               slope = 0.01,
                                               num_samples = self.num_samples,
                                               device = self.device,
                                            #    init_include = True
                                               )  # S1 cf

                        # Compute Q = p(y_t | tilde(X)^S_{t-n:t})
                        p_y_hat = self._model_predict(
                            x_rep_in_1.reshape(self.num_samples * batch_size, num_features, t + 1)
                        )


                        i_ab_S1_sample = self._compute_metric(p_y_exp, p_y_hat).reshape(
                            self.num_samples, batch_size
                        )
                        i_ab_S1 = torch.mean(i_ab_S1_sample, dim=0).detach().cpu().numpy()
                        i_ab_S1 = np.clip(i_ab_S1, -1e6, 1e6)
                        i_ab_S1_array[t, f, n, :] = i_ab_S1

                        ## Set S2 ##
                        # Now calculate for S2
                        # x_rep_in = self.replace_cfs(x[:, :, : t + 1], 
                        #                         t_start = t-n-1,
                        #                         t_end = t,
                        #                         CF = CF,
                        #                         mask_strategy = self.mask_strategy[0][1],
                        #                         d_start = f,
                        #                         height = self.height,
                        #                         slope = 0.01,
                        #                         num_samples = self.num_samples,
                        #                         device = self.device,
                        #                         # init_include = False
                        #                         )  # S2 cf
                        p_y_hat = self._model_predict(
                            x_rep_in_2.reshape(self.num_samples * batch_size, num_features, t + 1)
                        )
                        i_ab_S2_sample = self._compute_metric(p_y_exp, p_y_hat).reshape(
                            self.num_samples, batch_size
                        )
                        i_ab_S2 = torch.mean(i_ab_S2_sample, dim=0).detach().cpu().numpy()
                        i_ab_S2 = np.clip(i_ab_S2, -1e6, 1e6)
                        i_ab_S2_array[t, f, n, :] = i_ab_S2

                # Compute the I(S) array
                b = i_ab_S1_array[t, :, :, :] - i_ab_S2_array[t, :, :, :]
                IS_array[t, :, :, :] = b

                score = IS_array[t, :, ::-1, :].transpose(2, 0, 1)  # (bs, nfeat, time)

                # Pad the scores when time forward is less than window size.
                if score.shape[2] < self.window_size:
                    score = np.pad(score, ((0, 0), (0, 0), (self.window_size - score.shape[2], 0)))
                scores.append(score)
            self.log.info(f"Importance scoring of the batch done: Time elapsed: {(time() - tic):.4f}")

            scores = np.stack(scores).transpose((1, 2, 0, 3))  # (bs, fts, ts, window_size)
            i_ab_S1_array = i_ab_S1_array.transpose((3, 1, 0, 2))  # (bs, fts, ts, window_size)
            i_ab_S2_array = i_ab_S2_array.transpose((3, 1, 0, 2))  # (bs, fts, ts, window_size)
            IS_array = IS_array.transpose((3, 1, 0, 2))  # (bs, fts, ts, window_size)

            print('attribution done, ' \
            'scores.shape, ' \
            'iSabThick_array.shape,  ' \
            'iSabThickwithoutInit_array.shape,  ' \
            'IS_array.shape', 
            scores.shape, 
            i_ab_S1_array.shape,  
            i_ab_S2_array.shape,  
            IS_array.shape)
            return scores, i_ab_S1_array, i_ab_S2_array, IS_array




    def replace_cfs(
        self,
        X_in: torch.Tensor,
        t_start: int,
        t_end:   int,
        CF: torch.Tensor,
        mask_strategy: str,
        d_start: int,
        height:  int,
        slope:   float,
        num_samples: int,
        device:  torch.device,
        # init_include: bool
    ) -> torch.Tensor:
        """
        X_in:  (B, F, T)
        Returns:
        X_out: (S, B, F, T), where S=num_samples,
                each a masked version of X_in with independent CF draws.
        """
        B, F, T = X_in.shape

        # 1) build single‐sample upper‐right mask M[d,t] in {0,1}

        M1, M2 = self.get_mask(mask_strategy=mask_strategy, D=F, T=T, t_start=t_start, t_end=t_end, d_start=d_start, height=height)
        mask1 = torch.from_numpy(M1).to(device).bool().unsqueeze(0).expand(B, F, T)     # B x F x T                                       # F x T
        mask2 = torch.from_numpy(M2).to(device).bool().unsqueeze(0).expand(B, F, T)     # B x F x T                                       # F x T


        # 2) prepare expanded X and mask for batch of S samples
        X_exp = X_in.unsqueeze(0).expand(num_samples, B, F, T)  # (S, B, F, T)
        mask1_exp = mask1.unsqueeze(0).expand(num_samples, B, F, T)
        mask2_exp = mask2.unsqueeze(0).expand(num_samples, B, F, T)



        # 4) apply mask: wherever mask==1, use CF, else keep X
        X1_out = torch.where(mask1_exp, CF, X_exp).permute(2,0,1,3)  # (S, B, F, T)
        X2_out = torch.where(mask2_exp, CF, X_exp).permute(2,0,1,3)  # (S, B, F, T)

        return X1_out, X2_out



    def get_mask(self, mask_strategy, D, T, t_start, t_end, d_start, height):
        mask_strategy = mask_strategy[0]
        if (mask_strategy == "upper_triangular"):
            M1, M2 = self.fast_upper_right_mask(D, T, t_start, t_end, d_start, height)
        elif (mask_strategy == "upper_triangular_wo_head"):
            M1, M2 = self.build_upper_right_mask(D, T, t_start, t_end, d_start, height)
        elif (mask_strategy == "box"):
            M1, M2 = self.build_rectangular_mask(D, T, t_start, t_end, d_start, height)
        elif (mask_strategy == "box_wo_head"):
            M1, M2 = self.build_rectangular_mask(D, T, t_start, t_end, d_start, height)
        return M1, M2
        
    def build_rectangular_mask(
        self,
        D: int,
        T: int,
        t_start: int,
        t_end:   int,
        d_start: int,
        height:  int = 1
    ): 
        start = time_og.perf_counter()       # ← start timer

        M = np.zeros((D, T), dtype=int)
        row = np.zeros(D, dtype=np.uint8)
        col = np.zeros(T, dtype=np.uint8)
        
        width = t_start - t_end

        if (t_end >= T):
            t_end = T
        
        d_end = d_start + height
        if d_end > D:
            d_end = D

        # M[d_start:d_end, t_start:t_end] = 1
        # mark the intervals
        row[d_start:d_end] = 1
        col[t_start:t_end] = 1

        # 2) Outer product: only the rectangle becomes 1, the rest stays 0
        M = row[:, None] * col[None, :]

        M1 = M.copy()
        M1[d_start, t_start] = 0


        elapsed = time_og.perf_counter() - start   # ← end timer
        # print(f"[build_rectangular_mask] D={D},T={T},ts={t_start},te={t_end} took {elapsed:.6f}s")

        return M, M1


    def build_upper_right_mask(
        self,
        D: int,
        T: int,
        t_start: int,
        t_end:   int,
        d_start: int,
        height:  int = 1,
        slope:   float = 1.0
    ) -> np.ndarray:
        """
        Build a mask M of shape (D, T) as follows:
        1) Compute, for each t in [t_start..t_end], a row index
            r(t) = d_start + round( ((t-t_start)/(t_end-t_start))**slope * (height-1) ).
        2) For each (r, t) along that path, set
            M[0:r+1,  t : t_end+1] = 1
            i.e. fill the “upper‐right” triangle anchored at (r, t).
        3) Leave everything else as 0.

        Guarantees:
        - M[d_start,      t_start] = 1
        - M[d_start+height-1, t_end] = 1
        """
        M = np.zeros((D, T), dtype=int)
        span = t_end - t_start

        if (height + d_start >= D):
            height = D - d_start

        if height <= 1 or span <= 0:
            # just fill upper triangle from the single row
            for t in range(t_start, t_end+1):
                M[d_start:d_start+1, t:t_end+1] = 1
            
            M[d_start,      t_start] = 1
            M1 = M.copy()
            M1[d_start,      t_start] = 0

            # self.print_mask(M)
            return M, M1

        # Loop over each time in the interval
        for t in range(t_start, t_end+1):
            alpha = (t - t_start) / span
            alpha_s = alpha ** slope
            r = d_start + int(round(alpha_s * (height - 1)))

            # fill rows 0..r and cols t..t_end
            M[d_start:r+1, t:t_end+1] = 1

        # ensure endpoints

        M[d_start,      t_start] = 1
        M1 = M.copy()
        M1[d_start,      t_start] = 0

        M[d_start+height-1, t_end]   = 1

        # self.print_mask(M)
        return M, M1


    def fast_upper_right_mask(
        self,
        D: int,
        T: int,
        t_start: int,
        t_end:   int,
        d_start: int,
        height:  int = 1
        # slope:   float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast O(D·T) construction of the “upper-right” mask:

        Returns (M, M_minus) where each is a (D×T) array of 0/1,
        and M_minus is identical to M except M_minus[d_start, t_start] = 0.
        """
        # Clamp height so we don’t overflow
        height = min(height, D - d_start)
        span = t_end - t_start
        slope = 0.01
        # Trivial case: single row or no span
        if height <= 1 or span <= 0:
            M = np.zeros((D, T), dtype=int)
            M[d_start, t_start:t_end+1] = 1
            M[d_start, t_start] = 1
            M_minus = M.copy()
            M_minus[d_start, t_start] = 0
            return M, M_minus

        # 1) Build difference array of size (D+1)×(T+1)
        Diff = np.zeros((D+1, T+1), dtype=int)

        # 2) For each pivot column, mark the rectangle corners in O(1)
        for t in range(t_start, t_end + 1):
            alpha   = (t - t_start) / span
            alpha_s = alpha ** slope
            r_t     = d_start + int(round(alpha_s * (height - 1)))

            # increment the [d_start..r_t] x [t..t_end] block
            Diff[d_start,    t       ] += 1
            Diff[r_t + 1,    t       ] -= 1
            Diff[d_start,    t_end+1 ] -= 1
            Diff[r_t + 1,    t_end+1 ] += 1

        # 3) Run 2D prefix-sum over Diff to recover counts in [0..D-1,0..T-1]
        for i in range(D):
            for j in range(T):
                if i > 0:
                    Diff[i, j] += Diff[i-1, j]
                if j > 0:
                    Diff[i, j] += Diff[i, j-1]
                if i > 0 and j > 0:
                    Diff[i, j] -= Diff[i-1, j-1]

        # 4) Threshold to binary mask
        M = (Diff[:D, :T] > 0).astype(int)

        # 5) Enforce the exact endpoints
        M[d_start,        t_start] = 1
        M[d_start+height-1, t_end  ] = 1

        # 6) Build M_minus by zeroing the start cell
        M_minus = M.copy()
        M_minus[d_start, t_start] = 0
        
        self.print_mask(M)
        self.print_mask(M_minus)

        return M, M_minus

    def print_mask(self, M: np.ndarray):
        """Print the mask matrix of 0/1."""
        D, T = M.shape
        print("Mask (D×T):")
        for r in range(D):
            print("".join(str(x) for x in M[r]))
        print()



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

    def _init_generators(self):
        if self.joint:
            gen_path = self.path / "joint_generator"
            gen_path.mkdir(parents=True, exist_ok=True)
            self.generators = JointFeatureGenerator(
                self.num_features,
                self.device,
                gen_path,
                hidden_size=self.num_features * 3,
                prediction_size=self.window_size,
                data=self.data_name,
            )
        else:
            gen_path = self.path / "feature_generator"
            gen_path.mkdir(parents=True, exist_ok=True)
            self.generators = FeatureGenerator(
                self.num_features,
                self.device,
                gen_path,
                hidden_size=50,
                prediction_size=self.window_size,
                conditional=self.conditional,
                data=self.data_name,
            )

    def train_generators(
        self, train_loader, valid_loader, num_epochs=300
    ) -> GeneratorTrainingResults:
        self._init_generators()
        return self.generators.train_generator(train_loader, valid_loader, num_epochs)

    def test_generators(self, test_loader) -> float:
        test_loss = self.generators.test_generator(test_loader)
        self.log.info(f"Generator Test MSE Loss: {test_loss}")
        return test_loss

    def load_generators(self) -> None:
        self._init_generators()
        self.generators.load_generator()

    def _generate_counterfactuals(
        self, time_forward: int, x_in: torch.Tensor, x_current: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate the counterfactuals.

        Args:
            time_forward:
                Number of timesteps of counterfactuals we wish to generate.
            x_in:
                The past Tensor. Shape = (batch_size, num_features, num_times)
            x_current:
                The current Tensor if a conditional generator is used.
                Shape = (batch_size, num_features, time_forward). If the generator is not
                conditional, x_current is None.

        Returns:
            Counterfactual of shape (num_features, num_samples, batch_size, time_forward)

        """
        # x_in shape (bs, num_feature, num_time)
        # x_current shape (bs, num_feature, time_forward)
        # return counterfactuals shape (num_feature, num_samples, batchsize, time_forward)
        batch_size, _, num_time = x_in.shape
        if self.data_distribution is not None:
            # Random sample instead of using generator
            counterfactuals = torch.zeros(
                (self.num_features, self.num_samples, batch_size, time_forward), device=self.device
            )
            for f in range(self.num_features):
                values = self.data_distribution[:, f, :].reshape(-1)
                counterfactuals[f, :, :, :] = torch.Tensor(
                    self.rng.choice(values, size=(self.num_samples, batch_size, time_forward)),
                    device=self.device,
                )
            return counterfactuals

        if isinstance(self.generators, FeatureGenerator):
            mu, std = self.generators.forward(x_current, x_in, deterministic=True)
            mu = mu[:, :, :time_forward]
            std = std[:, :, :time_forward]  # (bs, f, time_forward)
            counterfactuals = mu.unsqueeze(0) + torch.randn(
                self.num_samples, batch_size, self.num_features, time_forward, device=self.device
            ) * std.unsqueeze(0)
            return counterfactuals.permute(2, 0, 1, 3)

        if isinstance(self.generators, JointFeatureGenerator):
            counterfactuals = torch.zeros(
                (self.num_features, self.num_samples, batch_size, time_forward), device=self.device
            )
            for f in range(self.num_features):
                mu_z, std_z = self.generators.get_z_mu_std(x_in)
                gen_out, _ = self.generators.forward_conditional_multisample_from_z_mu_std(
                    x_in,
                    x_current,
                    list(set(range(self.num_features)) - {f}),
                    mu_z,
                    std_z,
                    self.num_samples,
                )
                # gen_out shape (ns, bs, num_feature, time_forward)
                counterfactuals[f, :, :, :] = gen_out[:, :, f, :]
            return counterfactuals

        raise ValueError("Unknown generator or no data distribution provided.")

    def get_name(self):
        builder = ["biwinit", "window", str(self.window_size)]
        if self.num_samples != 3:
            builder.extend(["samples", str(self.num_samples)])
        if self.conditional:
            builder.append("cond")
        if self.joint:
            builder.append("joint")
        if self.height >= 2:
            builder.extend(["height", str(self.height)])
        builder.append(self.metric)
        if self.data_distribution is not None:
            builder.append("usedatadist")
        return "_".join(builder)
