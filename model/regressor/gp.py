from torch import Tensor
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_model


class GPRegressor(SingleTaskGP):
    
    def get_state_dict(self):
        return self.state_dict()


def get_gp_regressor(X: Tensor, Y: Tensor):
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=X.shape[-1], lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = GPRegressor(X, Y.unsqueeze(-1), covar_module=covar_module, 
                        likelihood=likelihood, outcome_transform=Standardize(m=1) )
    return model


def fit_gp_regressor(X: Tensor, Y: Tensor, *args, **kwargs) -> GPRegressor:
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=X.shape[-1], lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = GPRegressor(X, Y.unsqueeze(-1), covar_module=covar_module, 
                        likelihood=likelihood, outcome_transform=Standardize(m=1) )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    with gpytorch.settings.max_cholesky_size(float("inf")):
        fit_gpytorch_model(mll)
    return model
