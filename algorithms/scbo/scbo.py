from typing import Tuple
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
import gpytorch
from botorch.models import SingleTaskGP
import numpy as np

from ..base import ConstrainedBO
from model import fit_regressor, fit_classifier
from .transform import gaussian_copula, bilog
from .turbo import TurboState, update_state
from .sampling import ConstrainedMaxPosteriorSampling


class SCBO(ConstrainedBO):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state = None
        self.use_gaussian_copula = False

    def fit_surrogate(self, X: Tensor, Y: Tensor, C: Tensor, save_cache: bool = False) -> Tuple[SingleTaskGP, list]:
        
        is_valid = (C >= 0.0).all(dim=-1)
        if self.use_gaussian_copula:
            Y_valid = gaussian_copula(Y[is_valid], maximize=True)
        else:
            Y_valid = Y[is_valid]

        model = fit_regressor[self.reg_type](X[is_valid], Y_valid, model=self.cached_model)
        C = (C >= 0.0).to(float)
        if self.cached_constraint_models is not None:
            constraint_models = [fit_classifier[self.cls_type](X, Ci, model_i) for Ci, model_i in zip(C.T,self.cached_constraint_models)]
        else:
            constraint_models = [fit_classifier[self.cls_type](X, Ci) for Ci in C.T]

        if save_cache:
            self.cached_model, self.cached_constraint_models = model, constraint_models
        return model, constraint_models

    def optimize(self, X: Tensor, Y: Tensor, C: Tensor, load_cache: bool = False) -> Tensor:

        dim = X.shape[-1]
        if self.state is None:
            self.state = TurboState(dim, batch_size=1)
        else:
            self.state = update_state(self.state, Y[-1:], C[-1:])

        is_valid = (C >= 0.0).all(dim=-1)
        if self.use_gaussian_copula:
            Y_valid = gaussian_copula(Y[is_valid], maximize=True)
        else:
            Y_valid = Y[is_valid]

        if load_cache:
            model, constraint_models = self.cached_model, self.cached_constraint_models
        else:
            model, constraint_models = self.fit_surrogate(X, Y, C)

        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[is_valid][Y_valid.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.state.length / 2.0, 0.0, 1.0)

        # Thompson Sampling w/ Constraints (SCBO) 
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, device=self.device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points using Constrained Max Posterior Sampling
        with gpytorch.settings.max_cholesky_size(float("inf")):
            constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(model=model, constraint_model=constraint_models, replacement=False)
            with torch.no_grad():
                X_next = constrained_thompson_sampling(X_cand, num_samples=1)

        return X_next

    def acquisition(self, X: Tensor, Y: Tensor, C: Tensor, X_new: Tensor, load_cache: bool = False) -> Tensor:

        if load_cache:
            model, constraint_models = self.cached_model, self.cached_constraint_models
        else:
            model, constraint_models = self.fit_surrogate(X, Y, C)

        F_new = model.posterior(X_new.unsqueeze(1)).mean
        C_new = []
        for c_model in constraint_models:
            c_posterior = c_model.posterior(X_new)
            c_posterior_mean = c_posterior.probs
            C_new.append(c_posterior_mean)
        C_new = torch.stack(C_new, dim=-1)

        valid_samples = (C_new >= 0.5).all(dim=-1)

        if valid_samples.any():
            A_new = F_new.clone().squeeze(1)
            A_new[~valid_samples] = F_new.min()
        else:
            A_new = -torch.clamp(0.5 - C_new, min=0).sum(dim=-1)

        return A_new


class SCBO_T(SCBO):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_gaussian_copula = True


class SCBO_Restart(SCBO):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_sobol_init = 10
        self.sobol_init_counter = 0
        self.sobol_points = None
        self.sobol_start_index = 0
        
    def optimize(self, X: Tensor, Y: Tensor, C: Tensor, load_cache: bool = False) -> Tensor:

        dim = X.shape[-1]
        is_valid = (C[self.sobol_start_index:] >= 0.0).all(dim=-1)

        if self.state is None:
            self.state = TurboState(dim, batch_size=1)
        else:
            if self.state.restart_triggered:
                if self.sobol_init_counter == self.n_sobol_init: # all sobol points evaluated
                    if not is_valid.any(): # no valid sobol points
                        sobol = SobolEngine(dim, scramble=True)
                        self.sobol_points = sobol.draw(self.n_sobol_init).to(device=self.device)
                        self.sobol_init_counter = 0
                        X_next = self.sobol_points[self.sobol_init_counter].unsqueeze(0) # NOTE: assume batch size = 1
                        self.sobol_init_counter += 1
                        return X_next
                    else: # valid sobol points, exit with new state
                        self.state = TurboState(dim, batch_size=1)
                        self.sobol_points = None
                        self.sobol_init_counter = 0
                else:
                    if self.sobol_points is None: # initialize sobol points
                        sobol = SobolEngine(dim, scramble=True)
                        self.sobol_points = sobol.draw(self.n_sobol_init).to(device=self.device)
                        self.sobol_start_index = len(X)
                    X_next = self.sobol_points[self.sobol_init_counter].unsqueeze(0) # NOTE: assume batch size = 1
                    self.sobol_init_counter += 1
                    return X_next
            else:
                self.state = update_state(self.state, Y[-1:], C[-1:])

        if self.use_gaussian_copula:
            Y_valid = gaussian_copula(Y[self.sobol_start_index:][is_valid], maximize=True)
        else:
            Y_valid = Y[self.sobol_start_index:][is_valid]
        X_valid = X[self.sobol_start_index:][is_valid]

        if load_cache:
            model, constraint_models = self.cached_model, self.cached_constraint_models
        else:
            model, constraint_models = self.fit_surrogate(X, Y, C)

        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X_valid[Y_valid.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.state.length / 2.0, 0.0, 1.0)

        # Thompson Sampling w/ Constraints (SCBO) 
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, device=self.device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points using Constrained Max Posterior Sampling
        with gpytorch.settings.max_cholesky_size(float("inf")):
            constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(model=model, constraint_model=constraint_models, replacement=False)
            with torch.no_grad():
                X_next = constrained_thompson_sampling(X_cand, num_samples=1)

        return X_next

    def acquisition(self, X: Tensor, Y: Tensor, C: Tensor, X_new: Tensor, load_cache: bool = False) -> Tensor:

        if self.state.restart_triggered:
            return torch.zeros(X_new.shape[0])
        else:
            return super().acquisition(X, Y, C, X_new, load_cache)


class SCBO_T_Restart(SCBO_Restart):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_gaussian_copula = True
