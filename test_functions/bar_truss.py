import torch
from torch import Tensor
import math
from .base import ConstrainedSyntheticTestFunction


class ThreeBarTrussDesign(ConstrainedSyntheticTestFunction):
    """
    (2.3.6 from A test-suite of non-convex constrained optimization problems from the real-world and some baseline results)
    """
    dim = 2
    _bounds = [(0, 1), (0, 1)]
    _optimal_value: float = 263.89584337647364
    name = 'Three Bar Truss Design'

    l = 100
    P = 2
    sigma = 2

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = self.l * (X[:, 1] + 2 * math.sqrt(2) * X[:, 0])
        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        cons = -torch.stack([
            X[:, 1] / (2 * X[:, 1] * X[:, 0] + math.sqrt(2) * X[:, 0] ** 2) * self.P - self.sigma,
            (X[:, 1] + math.sqrt(2) * X[:, 0]) / (2 * X[:, 1] * X[:, 0] + math.sqrt(2) * X[:, 0] ** 2) * self.P - self.sigma,
            1 / (X[:, 0] + math.sqrt(2) * X[:, 1]) * self.P - self.sigma,
        ], dim=1)
        return cons
