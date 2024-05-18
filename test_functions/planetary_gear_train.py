import torch
from torch import Tensor
import math
from .base import ConstrainedSyntheticTestFunction

def safe_divide(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0.
    '''
    if type(x1) == torch.Tensor:
        x1 = x1.double()
    else:
        x1 = torch.tensor(x1).double()
    if type(x2) == torch.Tensor:
        x2 = x2.double()
    else:
        x2 = torch.tensor(x2).double()

    if x1.dim() < x2.dim():
        x1 = x1.unsqueeze(0).expand_as(x2)
    elif x2.dim() < x1.dim():
        x2 = x2.unsqueeze(0).expand_as(x1)

    result = torch.zeros_like(x1)
    mask = x2 != 0
    result[mask] = x1[mask] / x2[mask]
    
    return result

class PlanetaryGearTrain(ConstrainedSyntheticTestFunction):
    """
    Book: Mechanical Design Optimization Using Advanced Optimization Techniques
    (4.3.13 Example 20: Design of Planetary Gear Train)

    Design variables:
    N1 number of teeth (integer)
    N2 number of teeth (integer)
    N3 number of teeth (integer)
    N4 number of teeth (integer)
    N5 number of teeth (integer)
    N6 number of teeth (integer)
    P number of planet (integer)
    m1 module of gear (specified discrete values)
    m2 module of gear (specified discrete values)

    The objective is to minimize the gear ratio errors
    """
    dim = 9
    _bounds = [(17, 110), (14, 58), (14, 46), (17, 104), (14, 46), (25, 200), (3, 5), (1.75, 3), (1.75, 3)]
    _optimal_value: float = 3.685919732441471
    name = 'Planetary Gear Train'

    i01 = 3.11
    i02 = 1.84
    i0r = -3.11

    dmax = 220
    eps = 0.5

    def evaluate_true(self, X: Tensor) -> Tensor:
        n1 = torch.clip(X[:, 0].round(), 17, 110)   # Max bound: dmax/m1_min - 2 * Np_min - 2
        n2 = torch.clip(X[:, 1].round(), 14, 58)    # Max bound: (dmax/m1_min - Ns_min - 2) / 2
        n3 = torch.clip(X[:, 2].round(), 14, 46)    # Max bound: (dmax/m2_min - Ns_min - 6.5 - 2 * eps) / 2
        n4 = torch.clip(X[:, 3].round(), 17, 104)   # Max bound: (dmax/m2_min - 2 * Np_min - 6.5 - 2 * eps)
        n5 = torch.clip(X[:, 4].round(), 14, 46)    # Max bound: (dmax/m2_min - Ns_min - 6.5 - 2 * eps) / 2
        n6 = torch.clip(X[:, 5].round(), 25, 200)
        p = torch.clip(X[:, 6].round(), 3, 5)
        m1 = torch.round(X[:, 7] * 4) / 4.0  # Round to the closest discrete values (1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
        m2 = torch.round(X[:, 8] * 4) / 4.0  # Round to the closest discrete values (1.75, 2.0, 2.25, 2.5, 2.75, 3.0)

        i1 = safe_divide(n6, n4)
        i2 = safe_divide(n6 * (n1 * n3 + n2 * n4), (n1 * n3 * (n6 - n4)))
        ir = safe_divide(n2 * n6, n1 * n3)
        f = torch.max(torch.max(torch.abs(i1 - self.i01), torch.abs(i2 - self.i02)), torch.abs(ir - self.i0r))
        
        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:

        n1 = torch.clip(X[:, 0].round(), 17, 110)   # Max bound: dmax/m1_min - 2 * Np_min - 2
        n2 = torch.clip(X[:, 1].round(), 14, 58)    # Max bound: (dmax/m1_min - Ns_min - 2) / 2
        n3 = torch.clip(X[:, 2].round(), 14, 46)    # Max bound: (dmax/m2_min - Ns_min - 6.5 - 2 * eps) / 2
        n4 = torch.clip(X[:, 3].round(), 17, 104)   # Max bound: (dmax/m2_min - 2 * Np_min - 6.5 - 2 * eps)
        n5 = torch.clip(X[:, 4].round(), 14, 46)    # Max bound: (dmax/m2_min - Ns_min - 6.5 - 2 * eps) / 2
        n6 = torch.clip(X[:, 5].round(), 25, 200)
        p = torch.clip(X[:, 6].round(), 3, 5)
        m1 = torch.round(X[:, 7] * 4) / 4.0  # Round to the closest discrete values (1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
        m2 = torch.round(X[:, 8] * 4) / 4.0  # Round to the closest discrete values (1.75, 2.0, 2.25, 2.5, 2.75, 3.0)

        beta = torch.acos(torch.clip(safe_divide((n6 - n3)**2 + (n4 + n5)**2 - (n3 + n5)**2, 2 * (n6 - n3) * (n4 + n5)), -1, 1))


        # Note: here c1, c2... < 0 for the constraint to be satisfied
        c1 = m2 * (n6 + 2.5) - self.dmax
        c2 = m1 * (n1 + n2) + m1 * (n2 + 2) - self.dmax
        c3 = m2 * (n4 + n5) + m2 * (n5 + 2) - self.dmax
        c4 = torch.abs(m1 * (n1 + n2) - m2 * (n6 - n3)) - m1 - m2
        c4 = - ((n1 + n2) * torch.sin(safe_divide(math.pi, p)) - n2 -2 - self.eps)
        c5 = - ((n6 - n3) * torch.sin(safe_divide(math.pi, p)) - n3 -2 - self.eps)
        c6 = - ((n4 + n5) * torch.sin(safe_divide(math.pi, p)) - n5 -2 - self.eps)
        c7 = - ((n6 - n3)**2 + (n4 + n5)**2 - 2 * (n6 - n3) * (n4 + n5) * torch.cos(safe_divide(2 * math.pi, p) - beta) - (n3 + n5 + 2 + self.eps)**2)
        c8 = - (n6 - 2 * n3 - n4 - 4 - 2 * self.eps)
        c9 = - (n6 - n4 - 2 * n5 - 4 - 2 * self.eps)


        cons = -torch.stack([c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1)
        return cons
