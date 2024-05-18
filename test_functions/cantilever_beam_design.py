import math
import torch
from torch import Tensor
from typing import Optional
from .base import ConstrainedSyntheticTestFunction


class CantileverBeamDesign(ConstrainedSyntheticTestFunction):
    """
    Cheng, G.H., Gjernes, T. and Gary Wang, G., 2018. 
    An adaptive aggregation-based approach for expensively constrained black-box optimization problems. 
    Journal of Mechanical Design, 140(9), p.091402.
    """
    dim = 30
    b_bounds = [(0.01, 0.05)] * 10
    h_bounds = [(0.30, 0.65)] * 10
    l_bounds = [(0.50, 1.00)] * 10
    name = "Cantilever Beam Design"
    _optimal_value: float = 0.015730543730546942

    # Combine them into a single list
    _bounds = b_bounds + h_bounds + l_bounds

    def __init__(
        self, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        super().__init__(noise_std=noise_std, negate=negate)
        self.P = 50e3
        self.E = 200e9
        self.sigma_allow = 35e7
        self.AR = 25 
        self.L_min = 6
        self.d = 10
    
    def evaluate_true(self, X: Tensor) -> Tensor:
        # X is a tensor with 30 elements corresponding to b1, h1, l1, b2, h2, l2, ..., b10, h10, l10
        
        delta = 0
        for i in range(1, self.d + 1):
            bi = X[:, i - 1]  # b_i
            hi = X[:, 10 + i - 1]  # h_i
            li = X[:, 20 + i - 1]  # l_i
            
            # sum of l_j from j=i to d
            sum1 = X[:, 20 + i - 1:].sum(dim=1) if i <= self.d else 0
            # sum of l_j from j=i+1 to d
            sum2 = X[:, 20 + i:].sum(dim=1) if i <= self.d else 0

            # part inside the brackets for each i
            bracket_term = (12 / (bi * (hi**3))) * (sum1**3 - sum2**3)
            delta += bracket_term     
        
        f = self.P * delta / (3 * self.E)
        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        # X is a tensor with 30 elements corresponding to b1, h1, l1, b2, h2, l2, ..., b10, h10, l10

        cons = []
        sum_l = 0
        for i in range(1, self.d + 1):
            bi = X[:, i - 1]
            hi = X[:, 10 + i - 1]
            li = X[:, 20 + i - 1]
            # sum of l_j from j=i to d
            sum_lj = X[:, 20 + i - 1:].sum(dim=1) if i <= self.d else 0
            
            cons.append(((6 * self.P * sum_lj) / (bi * (hi**2)) - self.sigma_allow).unsqueeze(-1))  # Constraint 1(x10), bending stress
            cons.append((hi/bi - self.AR).unsqueeze(-1))    # Constraint 2(x10), cross section
            sum_l += li
        cons.append((self.L_min - sum_l).unsqueeze(-1)) # Constraint 3(x1), minimum length

        cons = -torch.cat(cons, dim=1)
        return cons
