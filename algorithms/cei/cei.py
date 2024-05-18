import torch
from torch import Tensor
from botorch.optim import optimize_acqf

from ..base import ConstrainedBO
from .acq import CEIAcq


class CEI(ConstrainedBO):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_restarts = 10
        self.raw_samples = 512

    def optimize(self, X: Tensor, Y: Tensor, C: Tensor, load_cache: bool = False) -> Tensor:

        is_valid = (C >= 0).all(dim=-1)
        assert is_valid.any()
        dim = X.shape[-1]

        if load_cache:
            model, constraint_models = self.cached_model, self.cached_constraint_models
        else:
            model, constraint_models = self.fit_surrogate(X, Y, C)

        acq = CEIAcq(
            model=model,
            constraint_model=constraint_models,
            best_f=Y[is_valid].max())

        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.tensor([[0.0] * dim, [1.0] * dim], device=self.device),
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={'batch_limit': 5, 'maxiter': 200},
        )

        X_next = candidates.detach()
        return X_next

    def acquisition(self, X: Tensor, Y: Tensor, C: Tensor, X_new: Tensor, load_cache: bool = False) -> Tensor:
        
        is_valid = (C >= 0).all(dim=-1)
        assert is_valid.any()

        if load_cache:
            model, constraint_models = self.cached_model, self.cached_constraint_models
        else:
            model, constraint_models = self.fit_surrogate(X, Y, C)

        acq = CEIAcq(
            model=model,
            constraint_model=constraint_models,
            best_f=Y[is_valid].max())

        A_new = acq(X_new.unsqueeze(1))
        return A_new
