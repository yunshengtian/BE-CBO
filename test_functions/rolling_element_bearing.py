import torch
from torch import Tensor
import math
from .base import ConstrainedSyntheticTestFunction


class RollingElementBearing(ConstrainedSyntheticTestFunction):
    """
    (2.3.14 from A test-suite of non-convex constrained optimization problems from the real-world and some baseline results)
    """
    dim = 10
    _bounds = [(125, 150), (10.5, 31.5), (4, 50), (0.515, 0.6), (0.515, 0.6), (0.4, 0.5), (0.6, 0.7), (0.3, 0.4), (0.02, 0.1), (0.6, 0.85)] # Dm, Db, Z, fi, fo, KDmin, KDmax, eps, e, xi
    name = "Rolling Element Bearing Design"
    _optimal_value: float = -81843.3

    D = 160
    d = 90
    Bw = 30

    def _calc_fc(self, Db, Dm, fi, fo):
        gamma = Db / Dm
        fc = 37.91 * (1 + (1.04 * ((1 - gamma) / (1 + gamma)) ** 1.72 * ((fi * (2 * fo - 1)) / (fo * (2 * fi - 1))) ** 0.41) ** (10 / 3)) ** -0.3 * \
            ((gamma ** 0.3 * (1 - gamma) ** 1.39) / (1 + gamma) ** (1 / 3)) * (2 * fi / (2 * fi - 1)) ** 0.41
        return fc

    def _calc_phi0(self, Db):
        D, d, Bw = self.D, self.d, self.Bw
        T = D - d - 2 * Db
        phi0 = 2 * math.pi - 2 * torch.acos(
            (((D - d) / 2 - 3 * (T / 4)) ** 2 + (D / 2 - T / 4 - Db) ** 2 - (d / 2 + T / 4) ** 2) /
            (2 * ((D - d) / 2 - 3 * (T / 4)) * (D / 2 - (T / 4) - Db))
        )
        return phi0

    def evaluate_true(self, X: Tensor) -> Tensor:
        Dm, Db, Z, fi, fo, KDmin, KDmax, eps, e, xi = X.T
        fc = self._calc_fc(Db, Dm, fi, fo)
        case_1 = Db <= 25.4
        case_2 = ~case_1
        Cd = torch.zeros(X.shape[0], dtype=X.dtype)
        Cd[case_1] = (fc * Z ** (2 / 3) * Db ** 1.8)[case_1]
        Cd[case_2] = (3.647 * fc * Z ** (2 / 3) * Db ** 1.4)[case_2]
        return -Cd

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        D, d, Bw = self.D, self.d, self.Bw
        Dm, Db, Z, fi, fo, KDmin, KDmax, eps, e, xi = X.T
        phi0 = self._calc_phi0(Db)
        cons = -torch.stack([
            Z - phi0 / (2 * torch.asin(Db / Dm)) - 1,
            KDmin * (D - d) - 2 * Db,
            2 * Db - KDmax * (D - d),
            xi * Bw - Db,
            0.5 * (D + d) - Dm,
            Dm - (0.5 + e) * (D + d),
            eps * Db - 0.5 * (D - Dm - Db),
        ], dim=1)
        return cons
