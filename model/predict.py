from typing import Union
from torch import Tensor
import torch


def predict_classifier(model, X: Tensor, mean_only: bool = True) -> Union[Tensor, tuple]:
    with torch.no_grad():
        latent = model.latent(X)
        pred = model.likelihood(latent)
        if mean_only:
            return pred.probs
        else:
            return pred.probs, latent.variance


def predict_regressor(model, X: Tensor, mean_only: bool = True) -> Union[Tensor, tuple]:
    with torch.no_grad():
        pred = model.posterior(X)
        if mean_only:
            return pred.mean
        else:
            return pred.mean, pred.variance
