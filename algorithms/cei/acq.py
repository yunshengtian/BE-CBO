from __future__ import annotations

from typing import Any, Optional, Union, List, Callable

import torch
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal


class CEIAcq(ExpectedImprovement):

    def __init__(
        self,
        model: Model,
        constraint_model: ModelListGP,
        best_f: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            best_f=best_f,
            posterior_transform=posterior_transform,
        )
        self.constraint_model = constraint_model

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean

        feasible_prob = 1.0
        for c_model in self.constraint_model:
            c_posterior = c_model.posterior(X)
            feasible_prob *= c_posterior.probs.unsqueeze(-1)
        feasible_prob = feasible_prob.squeeze(-1).squeeze(-1)

        view_shape = mean.shape[:-2]
        mean = mean.view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        cei = ei * feasible_prob
        return cei
