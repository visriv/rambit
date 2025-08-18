import torch
from torch.distributions import Bernoulli
from typing import Tuple, Optional
from src.explainer.explainers import BaseExplainer


class JIMEx(BaseExplainer):
    def __init__(
        self,
        num_masks: int = 3,
        wt_bounds: Tuple[int, int] = (1, 3),
        wd_bounds: Tuple[int, int] = (1, 3),
        device: Optional[torch.device] = None,
        metric: str = "pd",
        window_size: int = 10

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
        for t in range(T):
            for d in range(D):
                I_all[:, t, d] = self.attribute_one_cell(X, t, d)  # returns (B,)
        return I_all.permute(0,2,1).detach().cpu().numpy() # return in B x D x T shape



    def attribute_one_cell(
        self,
        X: torch.Tensor,
        source_t: int,
        source_d: int
    ) -> torch.Tensor:
        """
        Compute JIMEx attribution for cell (source_t, source_d) over a batch.
        Returns a (B,) tensor with one score per item in the batch.
        """
        DEBUG = False  # set False to silence prints

        X = X.to(self.device)
        B, D, T = X.shape

        if DEBUG:
            try:
                print(f"[JIMEx] X shape BxDxT={X.shape}, device={self.device}")
                print(f"[JIMEx] source: t={source_t}, d={source_d}")
                print(f"[JIMEx] L={self.L}, window_size={self.window_size}, "
                    f"Wt∈[{self.Wt_min},{self.Wt_max}], Wd∈[{self.Wd_min},{self.Wd_max}]")
            except Exception:
                # Attributes may not exist; keep running silently.
                pass

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

            if DEBUG and _ < 2:  # print details for first 2 masks
                zeros_M = (M == 0).sum().item()
                try:
                    print(f"[mask {_}] Wt={Wt}, Wd={Wd}, t0={t0}, d0={d0}, "
                        f"t_end={t0+Wt-1}, d_end={d0+Wd-1}")
                    print(f"[mask {_}] zeros in M: {zeros_M}/{M.numel()}")
                    print(f"[mask {_}] M1(src)={float(M1[0, source_d, source_t])}, "
                        f"M2(src)={float(M2[0, source_d, source_t])}")
                except Exception:
                    pass

                # local peek around the source cell: rows=d, cols=t
                t_lo = max(0, t0-1)
                t_hi = min(T, t0 + Wt + 1)
                d_lo = max(0, d0-1)
                d_hi = min(D, d0 + Wd + 1)

                local_M1 = M1[0, d_lo:d_hi, t_lo:t_hi].detach().to("cpu")
                local_M2 = M2[0, d_lo:d_hi, t_lo:t_hi].detach().to("cpu")

                # convert to int for cleaner view
                if DEBUG:
                    print(f"[mask {_}] M1 local[D {d_lo}:{d_hi}, T {t_lo}:{t_hi}] "
                        f"(rows=d, cols=t):\n{local_M1.int()}")
                    print(f"[mask {_}] M2 local[D {d_lo}:{d_hi}, T {t_lo}:{t_hi}] "
                        f"(rows=d, cols=t):\n{local_M2.int()}")

            all_M1.append(M1)
            all_M2.append(M2)
            t_max.append(t0 + Wt)

        # Now iterate over masks and obtain importance score based on prediction at t_target = t_max + w
        for l in range(self.L):
            I_cell_this_mask = 0
            count_valid_windows = 0

            if DEBUG and l < 2:
                t_first = t_max[l]
                t_last = min(T - 1, t_max[l] + self.window_size - 1)
                print(f"[score mask {l}] t_max={t_max[l]}, "
                    f"t_target range ≈ [{t_first}..{t_last}]")

            for w in range(1, self.window_size + 1):
                t_target = t_max[l] + w - 1
                if (t_target >= T):
                    continue

                count_valid_windows += 1
                # perturbed inputs
                X2 = X * all_M2[l]  # (B, D, T)
                X1 = X * all_M1[l]

                # model predictions (B,)
                with torch.no_grad():
                    f_orig = self._model_predict(X[:, :, 0:t_target + 1])   # shape [B]
                    f_masked1 = self._model_predict(X1[:, :, 0:t_target + 1])  # (B,)
                    f_masked2 = self._model_predict(X2[:, :, 0:t_target + 1])  # (B,)

                # metric per-sample (B,) then Δ
                i1 = self._compute_metric(f_orig, f_masked1)  # (B,)
                i2 = self._compute_metric(f_orig, f_masked2)  # (B,)

                I_cell_this_mask += (i1 - i2)

                if DEBUG and l < 2 and (w == 1 or w == self.window_size or t_target == T - 1):
                    # show summary stats only (means) to avoid spam
                    try:
                        print(f"[score mask {l}] w={w}, t_target={t_target}, "
                            f"mean(i1)={float(i1.mean()):.6f}, mean(i2)={float(i2.mean()):.6f}, "
                            f"mean(Δ)={float((i1 - i2).mean()):.6f}")
                    except Exception:
                        pass

            if (count_valid_windows == 0):
                I_cell += 0.0
                if DEBUG and l < 2:
                    print(f"[score mask {l}] no valid windows (t_target≥T).")
            else:
                I_cell += I_cell_this_mask / float(count_valid_windows)
                if DEBUG and l < 2:
                    try:
                        mean_mask_contrib = float((I_cell_this_mask / float(count_valid_windows)).mean())
                        print(f"[score mask {l}] averaged over {count_valid_windows} windows, "
                            f"mean contribution={mean_mask_contrib:.6f}")
                    except Exception:
                        pass

        # average over masks → (B,)
        I_cell = I_cell / float(self.L)

        if DEBUG:
            try:
                print(f"[JIMEx] Final I_cell shape: {tuple(I_cell.shape)} "
                    f"(B={B}); mean={float(I_cell.mean()):.6f}, "
                    f"min={float(I_cell.min()):.6f}, max={float(I_cell.max()):.6f}")
                # show a few samples (at most first 5)
                max_show = min(5, B)
                print(f"[JIMEx] I_cell[:{max_show}]: {I_cell[:max_show].detach().to('cpu')}")
            except Exception:
                pass

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
    
    def get_name(self):
        builder = ["jimex", "num_samples", str(self.L), "window_size", str(self.window_size)]

        builder.append(self.metric)
        # if self.data_distribution is not None:
        #     builder.append("usedatadist")
        return "_".join(builder)