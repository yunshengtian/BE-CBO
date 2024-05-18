import math
import torch
from torch import Tensor
from .base import ConstrainedSyntheticTestFunction


class GasTransmisionCompressor(ConstrainedSyntheticTestFunction):
    """
    Pant, M., Thangaraj, R. & Singh, V.P. 2009,
    "Optimization of Mechanical Design Problems Using Improved Differential Evolution Algorithm",
    International Journal of Recent Trends in Engineering, vol. 1, no. 5, pp. 21-25.
    """
    dim = 4
    _bounds = [(20, 50),(1, 10),(20, 50),(0.1,60)]
    _optimal_value = 2964895.417339161
    _optimizers = [(49.99999999999584, 1.178283949974269, 24.592589925270623, 0.3883530667669658)]
    name = 'Gas Transmission Compressor Design'

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = 861000 * X[:, 0]**(0.5) * X[:, 1] * X[:, 2]**(-2/3) * X[:, 3]**(-0.5) + 36900 * X[:, 2] + 772000000 * X[:, 0]**(-1) * X[:, 1]**(0.219) - 765430000 * X[:, 0]**(-1)
        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        cons = X[:, 3] * X[:, 1]**(-2) + X[:, 1]**(-2) - 1
        cons = -cons.unsqueeze(-1)
        return cons
