from typing import Optional, Union
import torch
from torch import Tensor
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP


class ConstrainedMaxPosteriorSampling(MaxPosteriorSampling):

    def __init__(
        self,
        model: Model,
        constraint_model: Union[ModelListGP, MultiTaskGP],
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
        minimize_constraints_only: bool = False,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under
                which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform.
            replacement: If True, sample with replacement.
            constraint_model: either a ModelListGP where each submodel
                is a GP model for one constraint function,
                or a MultiTaskGP model where each task is one
                constraint function
                All constraints are of the form c(x) <= 0.
                In the case when the constraint model predicts
                that all candidates violate constraints,
                we pick the candidates with minimum violation.
            minimize_constraints_only: False by default, if true,
                we will automatically return the candidates
                with minimum posterior constraint values,
                (minimum predicted c(x) summed over all constraints)
                reguardless of predicted objective values.
        """
        super().__init__(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            replacement=replacement,
        )
        self.constraint_model = constraint_model
        self.minimize_constraints_only = minimize_constraints_only

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor
                from which to sample (in the `N`
                dimension) according to the maximum
                posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim
            Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(X, observation_noise=observation_noise)
        samples = posterior.rsample(sample_shape=torch.Size([num_samples])) # num_samples x batch_shape x N x 1

        valid_samples = 1.0
        C_posterior_samples = []
        for c_model in self.constraint_model:
            c_latent = c_model.latent(X)
            c_latent_samples = c_latent.rsample(sample_shape=torch.Size([num_samples]))
            c_posterior_samples = c_model.likelihood(c_latent_samples).probs.unsqueeze(-1)
            valid_samples *= c_posterior_samples >= 0.5
            C_posterior_samples.append(c_posterior_samples)
        valid_samples = valid_samples.to(bool) # batch_shape x N x 1
        C_posterior_samples = torch.stack(C_posterior_samples) # n_constr x num_samples x batch_shape x N x 1

        if (valid_samples.sum() == 0) or self.minimize_constraints_only:
            # if none of the samples meet the constraints
            # we pick the one that minimizes total violation
            feasible_prob = C_posterior_samples.sum(dim=0).squeeze(-1) # num_samples x batch_shape x N
            idcs = torch.argmax(feasible_prob, dim=-1) # num_samples x batch_shape
            if idcs.ndim > 1:
                idcs = idcs.permute(*range(1, idcs.ndim), 0) # batch_shape x num_samples
            idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1)) # batch_shape x num_samples x d
            Xe = X.expand(*feasible_prob.shape[1:], X.size(-1)) # batch_shape x N x d
            return torch.gather(Xe, -2, idcs) # batch_shape x num_samples x d

        # replace all violators with -infinty so it will never choose them
        replacement_infs = -torch.inf * torch.ones(samples.shape).to(X.device).to(
            X.dtype
        )
        samples = torch.where(valid_samples, samples, replacement_infs) # num_samples x batch_shape x N x 1
        return self.maximize_samples(X, samples, num_samples) # batch_shape x num_samples x d
