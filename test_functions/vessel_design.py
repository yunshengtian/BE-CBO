import math
from typing import Optional
import torch
from torch import Tensor
from .base import ConstrainedSyntheticTestFunction


class VesselDesign(ConstrainedSyntheticTestFunction):
    """
    Hedar, AR., Fukushima, M. Derivative-Free Filter Simulated Annealing Method for Constrained Continuous Global Optimization. J Glob Optim 35, 521-549 (2006)
    """
    _optimal_value = 5868.764836
    dim = 4
    _bounds = [(0, 6.25), (0, 6.25), (10, 50), (150, 250)]
    _optimizers = [(0.768325709391, 0.379783796302, 39.809622248187, 207.225559518596)]
    name = 'Pressure Vessel Design'

    def __init__(
        self, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = 0.6224 * X[:, 0] * X[:, 2] * X[:, 3] + 1.7781 * X[:, 1] * X[:, 2].pow(2) + 3.1661 * X[:, 0].pow(2) * X[:, 3] + 19.84 * X[:, 0].pow(2) * X[:, 3]
        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        con1 = X[:, 0] - 0.0193 * X[:, 2]
        con2 = X[:, 1] - 0.00954 * X[:, 2]
        con3 = math.pi*X[:, 2].pow(2) * X[:, 3] + (4 * math.pi / 3) * X[:, 2].pow(3) - 1296000
        con4 = -X[:, 3] + 240
        cons = torch.stack([con1, con2, con3, con4], dim=1)
        return cons
