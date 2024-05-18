import torch
from torch import Tensor
from .base import ConstrainedSyntheticTestFunction


class TensionCompressionString(ConstrainedSyntheticTestFunction):
    """
    x1: N of active coils of the spring
    x2: D the winding diameter
    x3: d the wire diameter
    """
    dim = 3
    _bounds = [(2, 15), (0.25, 1.3), (0.05, 2)] # NOTE: x1 is integer
    _optimal_value = 0.012666
    _optimizers = [(11.329555, 0.356032, 0.051661), (11.6611924, 0.3505298, 0.0514305)]
    name = 'Tension/Compression String Design'

    def evaluate_true(self, X: Tensor) -> Tensor:
        X1 = torch.clip(X[:, 0].round(), 2, 15)
        f = (X1 + 2) * X[:, 1] * X[:, 2] ** 2
        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X1 = torch.clip(X[:, 0].round(), 2, 15)
        cons = torch.stack([
            X[:, 1] ** 3 * X1 / (71785 * X[:, 2] ** 4) - 1,
            1 - (4 * X[:, 1] ** 2 - X[:, 2] * X[:, 1]) / (12566 * (X[:, 1] * X[:, 2] ** 3 - X[:, 2] ** 4)) - 1 / (5108 * X[:, 2] ** 2),
            140.45 * X[:, 2] / (X[:, 1] ** 2 * X1) - 1,
            1 - (X[:, 1] + X[:, 2]) / 1.5,
        ], dim=1)
        return cons
