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
        metric: str = "pd"

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
        X = X.to(self.device)
        B, D, T = X.shape

        # 1) original prediction vector (B,)
        with torch.no_grad():
            f_orig = self._model_predict(X)  # shape [B], logits or probs as your _compute_metric expects

        # accumulate Δ per-sample
        I_cell = torch.zeros(B, device=self.device, dtype=X.dtype)

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

            # ensure target is kept in M2
            M[source_t, source_d] = 1.0

            # reshape to match X (B, D, T)
            M = M.permute(1, 0).unsqueeze(0)  # (1, D, T)

            # M2 keeps target bit; M1 flips it so M1/M2 differ at (t*,d*)
            M2 = M.clone()
            M1 = M.clone()
            M1[:, source_d, source_t] = 0.0

            # perturbed inputs
            X2 = X * M2  # (B, D, T)
            X1 = X * M1

            # model predictions (B,)
            with torch.no_grad():
                f_masked1 = self._model_predict(X1)  # (B,)
                f_masked2 = self._model_predict(X2)  # (B,)

            # metric per-sample (B,) then Δ
            i1 = self._compute_metric(f_orig, f_masked1)  # (B,)
            i2 = self._compute_metric(f_orig, f_masked2)  # (B,)
            I_cell += (i1 - i2)

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
    
    def get_name(self):
        builder = ["jimex", "num_samples", str(self.L)]

        builder.append(self.metric)
        # if self.data_distribution is not None:
        #     builder.append("usedatadist")
        return "_".join(builder)