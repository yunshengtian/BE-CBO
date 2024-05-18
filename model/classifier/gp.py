from torch import Tensor
import torch
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from botorch.models.gpytorch import GPyTorchModel
from botorch.fit import fit_gpytorch_model
from botorch.models.utils import gpt_posterior_settings


class BernoulliVariationalGP(ApproximateGP, GPyTorchModel):
    # see https://github.com/pytorch/botorch/issues/640
    
    def __init__(self, train_X, train_Y, kernel_type='rbf', ard=False, prior_constr=False):
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

        variational_distribution = CholeskyVariationalDistribution(train_X.size(0))
        variational_strategy = VariationalStrategy(
            self, train_X, variational_distribution
        )
        super(BernoulliVariationalGP, self).__init__(variational_strategy)
        
        kernel_kwargs = {
            'ard_num_dims': train_X.shape[-1] if ard else None,
            'lengthscale_constraint': Interval(0.005, 4.0) if prior_constr else None
        }

        self.mean_module = ConstantMean()
        if kernel_type == 'rbf':
            self.covar_module = ScaleKernel(RBFKernel(**kernel_kwargs))
        elif kernel_type == 'matern-3/2':
            self.covar_module = ScaleKernel(MaternKernel(nu=1.5, **kernel_kwargs))
        elif kernel_type == 'matern-5/2':
            self.covar_module = ScaleKernel(MaternKernel(nu=2.5, **kernel_kwargs))
        else:
            raise NotImplementedError
        self.likelihood = BernoulliLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPClassifier(BernoulliVariationalGP):

    def latent(self, X: Tensor):
        self.eval()
        with gpt_posterior_settings():
            latent = self(X)
        return latent

    def posterior(self, X: Tensor):
        latent = self.latent(X)
        posterior = self.likelihood(latent)
        return posterior

    def posterior_band(self, X: Tensor):
        latent = self.latent(X)
        latent_lower = latent.mean - latent.stddev
        latent_upper = latent.mean + latent.stddev
        return self.likelihood(latent_lower), self.likelihood(latent_upper)

    def get_state_dict(self):
        return self.state_dict()


def fit_gp_classifier(X: Tensor, Y: Tensor, *args, **kwargs) -> GPClassifier:
    assert X.device == Y.device, 'input should be on the same device!'
    model = GPClassifier(X, Y, **kwargs).to(X.device)
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, len(X))

    with gpytorch.settings.max_cholesky_size(float("inf")):
        fit_gpytorch_model(mll)
    return model
