import torch
from torch import Tensor
from .base import ConstrainedSyntheticTestFunction


class Simionescu(ConstrainedSyntheticTestFunction):
    """
    
    """
    dim = 2
    _bounds = [(-1.25, 1.25), (-1.25, 1.25)]
    _optimal_value = -0.072
    _optimizers = [(0.84852813, -0.84852813), (-0.84852813, 0.84852813)]
    name = 'Simionescu Function'

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = 0.1 * X[:, 0] * X[:, 1]
        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        rt, rs, n = 1, 0.2, 8
        cons = (rt + rs * torch.cos(n * torch.arctan(X[:, 0] / X[:, 1]))) ** 2 - X[:, 0] ** 2 - X[:, 1] ** 2
        cons = cons.unsqueeze(-1)
        return cons


class SimonescuShift(Simionescu):

    _optimal_value = -0.033333333333333215
    _optimizers = [(-0.6666666496101856, 0.66666669349225)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = 0.1 * X[:, 0] * X[:, 1] + 0.1 * (X[:, 0] - X[:, 1] + 1) ** 2
        return f
    