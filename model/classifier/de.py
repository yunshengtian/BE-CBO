import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.lazy import DiagLazyTensor
from botorch.models.gpytorch import GPyTorchModel
from botorch.fit import fit_gpytorch_model
from botorch.models.utils import gpt_posterior_settings

from ..fit import fit_gpytorch_torch # fixed memory issue


class DEClassifier(ApproximateGP, GPyTorchModel):

    def __init__(self, train_X, train_Y, n_network=5, hidden_layers=2, hidden_size=64) -> None:
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

        variational_distribution = CholeskyVariationalDistribution(train_X.size(0))
        variational_strategy = VariationalStrategy(
            self, train_X, variational_distribution
        )
        super(DEClassifier, self).__init__(variational_strategy)

        input_dim = train_X.shape[-1]
        # self.hid_dim = int(input_dim*64) # too big for high-dim problems
        self.hid_dim = int(np.log2(input_dim)) * hidden_size
        self.nets = []
        for _ in range(n_network):
            net = [nn.Linear(input_dim, self.hid_dim)]
            for _ in range(hidden_layers):
                net.append(nn.ReLU(inplace=True))
                net.append(nn.Linear(self.hid_dim, self.hid_dim))
            net.append(nn.ReLU(inplace=True))
            net.append(nn.Linear(self.hid_dim, 1))
            self.nets.append(nn.Sequential(*net))
        self.nets = nn.ModuleList(self.nets)
        self.net_keys = self.nets.state_dict().keys()
        self.softplus = nn.Softplus()
        self.likelihood = BernoulliLikelihood()

    def forward(self, x):
        output_x = []
        for net in self.nets:
            output_x.append(net(x).squeeze(-1))
        output_x = torch.stack(output_x)
        mean_x = output_x.mean(dim=0)
        var_x = self.softplus(output_x.var(dim=0))
        covar_x = DiagLazyTensor(var_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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
        return self.nets.state_dict()


def fit_de_classifier(X: Tensor, Y: Tensor, model = None, n_network=5, hidden_layers=2, hidden_size=64, lr=3e-4, **kwargs):
    retrain = True
    assert X.device == Y.device, 'input should be on the same device!'
    new_model = DEClassifier(X, Y, n_network=n_network, hidden_layers=hidden_layers, hidden_size=hidden_size).to(X.device)
    if not retrain and model is not None:
        state_dict = model.nets.state_dict()
        new_state_dict = {}
        for key in state_dict:
            if key in new_model.net_keys:
                new_state_dict[key] = state_dict[key]
        new_model.nets.load_state_dict(new_state_dict)
    model = new_model
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, len(X))

    with gpytorch.settings.max_cholesky_size(float("inf")):
        fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options={'disp': False, 'lr': lr, 'maxiter': 1000})

    return model
