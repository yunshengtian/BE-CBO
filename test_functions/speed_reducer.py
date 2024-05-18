import math
import torch
from torch import Tensor
from .base import ConstrainedSyntheticTestFunction


class SpeedReducerDesign(ConstrainedSyntheticTestFunction):
    """
    A. Lemonge, H. Barbosa, C. Borges, and F. Silva.
    Constrained optimization problems in mechanical
    engineering design using a real-coded steady-state
    genetic algorithm. Mecanica Computacional, 29:9287-
    9303, 2010.
    """
    dim = 7
    _bounds = [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), (7.8, 8.3), (2.9, 3.9), (5, 5.5)] # NOTE: x3 is integer
    _optimal_value = 2996.3482
    _optimizers = [(3.5, 0.7, 17, 7.3, 7.8, 3.350215, 5.286683)]
    name = 'Speed Reducer Design'

    def evaluate_true(self, X: Tensor) -> Tensor:
        X3 = torch.clip(X[:, 2].round(), 17, 28)
        f = 0.7854 * X[:, 0] * X[:, 1] ** 2 * (3.3333 * X3 ** 2 + 14.9334 * X3 - 43.0934) \
            - 1.508 * X[:, 0] * (X[:, 5] ** 2 + X[:, 6] ** 2) + 7.4777 * (X[:, 5] ** 3 + X[:, 6] ** 3) \
            + 0.7854 * (X[:, 3] * X[:, 5] ** 2 + X[:, 4] * X[:, 6] ** 2)
        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X3 = torch.clip(X[:, 2].round(), 17, 28)
        cons = torch.stack([
            1 - 27 * X[:, 0] ** (-1) * X[:, 1] ** (-2) * X3 ** (-1),
            1 - 397.5 * X[:, 0] ** (-1) * X[:, 1] ** (-2) * X3 ** (-2),
            1 - 1.93 * X[:, 1] ** (-1) * X3 ** (-1) * X[:, 3] ** (-3) * X[:, 5] ** (-4),
            1 - 1.93 * X[:, 1] ** (-1) * X3 ** (-1) * X[:, 4] ** (-3) * X[:, 6] ** (-4),
            1100 - ((745 * X[:, 3] / (X[:, 1] * X3)) ** 2 + 16.9 * 10 ** 6) ** 0.5 / (0.1 * X[:, 5] ** 3),
            850 - ((745 * X[:, 4] / (X[:, 1] * X3)) ** 2 + 157.5 * 10 ** 6) ** 0.5 / (0.1 * X[:, 6] ** 3),
            40 - X[:, 1] * X3,
            X[:, 0] / X[:, 1] - 5,
            12 - X[:, 0] / X[:, 1],
            1 - (1.5 * X[:, 5] + 1.9) * X[:, 3] ** (-1),
            1 - (1.1 * X[:, 6] + 1.9) * X[:, 4] ** (-1),
        ], dim=1)
        return cons
