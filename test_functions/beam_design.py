import math
import torch
from torch import Tensor
from .base import ConstrainedSyntheticTestFunction


class WeldedBeamDesign(ConstrainedSyntheticTestFunction):
    """
    Hedar, AR., Fukushima, M. Derivative-Free Filter Simulated Annealing Method for Constrained Continuous Global Optimization. J Glob Optim 35, 521-549 (2006)
    """
    dim = 4
    _bounds = [(0.125, 10), (0.1, 10), (0.1, 10), (0.1, 10)]
    _optimal_value = 2.381065
    _optimizers = [(0.24435257, 6.2157922, 8.2939046, 0.24435258)]
    name = 'Welded Beam Design'

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2, x3, x4 = X.T
        f = 1.10471 * x1 ** 2 * x2 + 0.04811 * x3 * x4 * (14 + x2)
        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        x1, x2, x3, x4 = X.T
        tou1 = 6000.0 / (math.sqrt(2) * x1 * x2)
        tou2 = 6000.0 * (14 + 0.5 * x2) * torch.sqrt(0.25 * (x2 ** 2 + (x1 + x3) ** 2)) / \
            (2 * 0.707 * x1 * x2 * (x2 ** 2 / 12 + 0.25 * (x1 + x3) ** 2))
        tou = torch.sqrt(tou1 ** 2 + tou2 ** 2 + x2 * tou1 * tou2 / torch.sqrt(0.25 * (x2 ** 2 + (x1 + x3) ** 2)))
        sigma = 504000.0 / (x3 ** 2 * x4)
        P_c = 64746.022 * (1 - 0.0282346 * x3) * x3 * x4 ** 3
        delta = 2.1952 / (x3 ** 3 * x4)
        cons = torch.stack([
            13000 - tou,
            30000 - sigma,
            x4 - x1,
            P_c - 6000,
            0.25 - delta,
        ], dim=1)
        return cons
