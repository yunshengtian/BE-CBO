import torch
from torch import Tensor
from .base import ConstrainedSyntheticTestFunction


class Townsend(ConstrainedSyntheticTestFunction):
    """
    
    """
    dim = 2
    _bounds = [(-2.25, 2.25), (-2.5, 1.75)]
    _optimal_value = -2.0239884
    _optimizers = [(2.0052938, 1.1944509)]
    name = 'Townsend Function'

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = -torch.cos((X[:, 0] - 0.1) * X[:, 1]) ** 2 - X[:, 0] * torch.sin(3 * X[:, 0] + X[:, 1])
        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        t = torch.atan2(X[:, 0], X[:, 1])
        cons = (2 * torch.cos(t) - 0.5 * torch.cos(2 * t) - 0.25 * torch.cos(3 * t) - 0.125 * torch.cos(4 * t)) ** 2 + (2 * torch.sin(t)) ** 2 - X[:, 0] ** 2 - X[:, 1] ** 2
        cons = cons.unsqueeze(-1)
        return cons


class TownsendShift(Townsend):

    _optimal_value = -3.6430629882311085
    _optimizers = [(1.667655240702945, -0.025073921844245728)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        shift = torch.Tensor([1, 0])
        X = X + shift
        f = -torch.cos((X[:, 0] - 0.1) * X[:, 1]) ** 2 - X[:, 0] * torch.sin(3 * X[:, 0] + X[:, 1])
        return f
