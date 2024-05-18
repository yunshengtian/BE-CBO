import torch
from torch import Tensor
from botorch.optim import optimize_acqf
import numpy as np
from torch.quasirandom import SobolEngine

from ..base import ConstrainedBO
from botorch.acquisition import ExpectedImprovement


def get_feasible_prob(constraint_model, X):
    '''
    Each callable is expected to take a `(num_restarts) x q x d`-dim tensor as an
    input and return a `(num_restarts) x q`-dim tensor with the constraint
    values.
    '''
    feasible_prob = 1.0
    for c_model in constraint_model:
        c_posterior = c_model.posterior(X.unsqueeze(0))
        feasible_prob *= c_posterior.probs
    return feasible_prob


def get_feasible_band(constraint_model, X):
    feasible_prob_lower, feasible_prob_upper = 1.0, 1.0
    for c_model in constraint_model:
        c_posterior_lower, c_posterior_upper = c_model.posterior_band(torch.atleast_2d(X))
        feasible_prob_lower *= c_posterior_lower.probs
        feasible_prob_upper *= c_posterior_upper.probs
    return feasible_prob_lower, feasible_prob_upper


def get_constr_lb_dynamic(constraint_model, X, lb):
    feasible_prob_lower, feasible_prob_upper = get_feasible_band(constraint_model, X)
    return feasible_prob_upper - lb


def find_boundary_sample(constraint_model, X):
    X = torch.tensor(X, requires_grad=True)
    optimizer = torch.optim.Adam([X], lr=1e-2)
    best_loss = np.inf
    stop_count = 0
    while True:
        loss = ((get_feasible_prob(constraint_model, X).squeeze() - 0.5) ** 2).mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        X.data = torch.clamp(X, 0, 1)
        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
        else:
            stop_count += 1
        if loss.item() < 0.02 or stop_count > 10:
            break
    return X.detach().clone()


class BECBO(ConstrainedBO):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_restarts = 10
        self.raw_samples = 512

    def get_acq(self, model, constraint_models, best_f):
        return ExpectedImprovement(
            model=model,
            best_f=best_f)

    def optimize(self, X: Tensor, Y: Tensor, C: Tensor, load_cache: bool = False) -> Tensor:

        is_valid = (C >= 0).all(dim=-1)
        assert is_valid.any()
        dim = X.shape[-1]

        if load_cache:
            model, constraint_models = self.cached_model, self.cached_constraint_models
        else:
            model, constraint_models = self.fit_surrogate(X, Y, C)

        acq = self.get_acq(model, constraint_models, Y[is_valid].max())

        constr_func = lambda XL: get_constr_lb_dynamic(constraint_models, XL, 0.5)

        # generate initial feasible samples
        iter_count = 0
        X_init = None
        while X_init is None or len(X_init) < 10:
            sobol = SobolEngine(dim, scramble=True)
            X_sample = sobol.draw(n=1000)
            prob_sample = get_feasible_prob(constraint_models, X_sample).squeeze()
            loss = (prob_sample - 0.5) ** 2
            X_best = torch.cat([
                X_sample[torch.argsort(loss)[:5]], 
                X_sample[torch.argsort(acq(X_sample.unsqueeze(1)), descending=True)[:5]]
            ])
            X_best = find_boundary_sample(constraint_models, X_best)
            constr_lb = constr_func(X_best)
            valid = (constr_lb >= 0)
            iter_count += 1
            if valid.any():
                if X_init is None:
                    X_init = X_best[valid].unsqueeze(1)
                else:
                    X_init = torch.cat([X_init, X_best[valid].unsqueeze(1)], dim=0)
                iter_count = 0
            if iter_count == 5:
                if X_init is None:
                    return torch.rand((1, X.shape[-1]))
                else:
                    break
        X_init = X_init[:10].detach()

        # optimize with hard boundary constraints
        try:
            candidates, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.tensor([[0.0] * dim, [1.0] * dim], device=self.device),
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                options={'batch_limit': 1, 'maxiter': 200},
                batch_initial_conditions=X_init,
                nonlinear_inequality_constraints=[constr_func]
            )
        except Exception as e:
            print(f'Exception from {self.fun.name} seed {self.seed} at iter {len(X)}:', str(e))
            raise e

        X_next = candidates.detach()
        return X_next

    def acquisition(self, X: Tensor, Y: Tensor, C: Tensor, X_new: Tensor, load_cache: bool = False) -> Tensor:
        
        is_valid = (C >= 0).all(dim=-1)
        assert is_valid.any()

        if load_cache:
            model, constraint_models = self.cached_model, self.cached_constraint_models
        else:
            model, constraint_models = self.fit_surrogate(X, Y, C)

        acq = self.get_acq(model, constraint_models, Y[is_valid].max())

        A_new = acq(X_new.unsqueeze(1))
        return A_new
    