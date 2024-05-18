from typing import Tuple
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from ..base import ConstrainedBO


class RandomSobol(ConstrainedBO):

    X_sample = None
    n_init = None

    def optimize(self, X: Tensor, Y: Tensor, C: Tensor, load_cache: bool = False) -> Tensor:

        if self.X_sample is None:
            dim = X.shape[-1]
            self.n_init = len(X)
            sobol = SobolEngine(dimension=dim, scramble=True)
            self.X_sample = sobol.draw(n=self.eval_budget - self.n_init).type(X.dtype)

        return self.X_sample[len(X) - self.n_init].unsqueeze(0)

    def predict(self, X: Tensor, Y: Tensor, C: Tensor, X_new: Tensor, load_cache: bool = False, mean_only: bool = True) -> Tuple[Tensor, Tensor]:

        Y_pred_mean, C_pred_mean = torch.zeros(X_new.shape[0], dtype=X.dtype), torch.zeros((X_new.shape[0], C.shape[1]), dtype=X.dtype)
        if mean_only:
            return Y_pred_mean, C_pred_mean
        else:
            Y_pred_var, C_pred_var = Y_pred_mean.clone(), C_pred_mean.clone()
            return Y_pred_mean, Y_pred_var, C_pred_mean, C_pred_var
