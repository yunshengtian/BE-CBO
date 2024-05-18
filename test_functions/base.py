from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.transforms import unnormalize


'''
Hints:
    evaluate_true: minimize
    evaluate_stack_true: >=0 is feasible, <0 is infeasible
'''


class ConstrainedBaseTestProblem(BaseTestProblem, ABC):
    r"""Base class for test functions with constraints.

    In addition to one or more objectives, a problem may have a number of outcome
    constraints of the form `c_i(x) >= 0` for `i=1, ..., n_c`.

    This base class provides common functionality for such problems.
    """

    _check_grad_at_opt: bool = False

    def evaluate_slack(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the constraint slack on a set of points.

        Constraints `i` is assumed to be feasible at `x` if the associated slack
        `c_i(x)` is positive. Zero slack means that the constraint is active. Negative
        slack means that the constraint is violated.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraint slacks: `c_1(X), ...., c_{n_c}(X)`.
            noise: If `True`, add observation noise to the slack as specified by
                `noise_std`.

        Returns:
            A `batch_shape x n_c`-dim tensor of constraint slack (where non-negative slack
                corresponds to the constraint being feasible).
        """
        cons = self.evaluate_slack_true(X=X)
        if noise and self.noise_std is not None:
            # TODO: Allow different noise levels for objective and constraints (and
            # different noise levels between different constraints)
            cons += self.noise_std * torch.randn_like(cons)
        return cons

    def is_feasible(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate whether the constraints are feasible on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraints.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim boolean tensor that is `True` iff all constraint
                slacks (potentially including observation noise) are non-negative.
        """
        return (self.evaluate_slack(X=X, noise=noise) >= 0.0).all(dim=-1)

    @abstractmethod
    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the constraint slack (w/o observation noise) on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraint slacks: `c_1(X), ...., c_{n_c}(X)`.

        Returns:
            A `batch_shape x n_c`-dim tensor of constraint slack (where non-negative slack
                corresponds to the constraint being feasible).
        """
        pass  # pragma: no cover


class ConstrainedSyntheticTestFunction(ConstrainedBaseTestProblem, SyntheticTestFunction):

    dim: int                                    # design space dimension (required)
    _bounds: List[Tuple[float, float]]          # design space bound (required)
    _optimizers: List[Tuple[float, ...]] = None # optimal designs
    _optimal_value: float = None                # optimal performance
    num_objectives: int = 1                     # performance space dimension
    _check_grad_at_opt: bool = False            # not sure how this works
    name: str = __name__

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Base constructor for constrained synthetic test functions.

        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    def forward(self, X: Tensor, noise: bool = False) -> Tuple[Tensor, Tensor]:
        r"""Evaluate the function on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim tensor ouf function evaluations.
            A `batch_shape x n_c`-dim tensor of constraint slack (where non-negative slack
                corresponds to the constraint being feasible).
        """
        f = super().forward(X=unnormalize(X, self.bounds), noise=noise)
        cons = super().evaluate_slack(X=unnormalize(X, self.bounds), noise=noise)
        cons = (cons >= 0.0).all(dim=-1).unsqueeze(-1).to(float) - 0.5
        return f, cons

    def is_feasible(self, X: Tensor, noise: bool = False) -> Tensor:
        r"""Evaluate whether the constraints are feasible on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraints.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim boolean tensor that is `True` iff all constraint
                slacks (potentially including observation noise) are non-negative.
        """
        return (self.evaluate_slack(X=unnormalize(X, self.bounds), noise=noise) >= 0.0).all(dim=-1)

    @property
    def optimal_value(self) -> float:
        r"""The global minimum (maximum if negate=True) of the function."""
        if self._optimal_value == None:
            return None
        return -self._optimal_value if self.negate else self._optimal_value

    def compute_optima(self, method='cma') -> Tuple[tuple, float]:
        if method == 'cma':
            r"""Compute the optimal design and performance empirically by evolutionary algorithm."""
            import cma

            x0 = np.random.rand(self.dim)
            y_bad = None
            es = cma.CMAEvolutionStrategy(x0=x0, sigma0=0.5, inopts={'bounds': [0, 1], 'popsize': 1000, 'verbose': -1})
            with torch.no_grad():
                while not es.stop():
                    xs = es.ask()
                    X = torch.tensor(np.array(xs))
                    Y, C = self.forward(X)
                    if self.negate:
                        if y_bad is None: y_bad = float(torch.min(Y))
                        ys = -np.maximum(Y.numpy() - y_bad, 0) * (C >= 0.0).all(dim=-1).float().numpy()
                    else:
                        if y_bad is None: y_bad = float(torch.max(Y))
                        ys = -np.maximum(y_bad - Y.numpy(), 0) * (C >= 0.0).all(dim=-1).float().numpy()
                    es.tell(xs, ys)
            
            best_x = unnormalize(es.best.x, self.bounds.numpy())
            best_x = tuple(best_x)
            if self.negate:
                best_y = es.best.f - y_bad
            else:
                best_y = es.best.f + y_bad
            return best_x, best_y
        
        elif method == 'random':
            r"""Compute the optimal design and performance empirically by random sampling."""
            X = torch.rand((int(1e6), self.dim))
            Y, C = self.forward(X)
            if self.negate:
                y_bad = float(torch.min(Y))
                YC = (Y - y_bad) * (C >= 0.0).all(dim=-1).to(Y.dtype)
            else:
                y_bad = float(torch.max(Y))
                YC = (y_bad - Y) * (C >= 0.0).all(dim=-1).to(Y.dtype)

            best_idx = torch.argmax(YC, dim=0).numpy()
            best_x = unnormalize(X[best_idx], self.bounds).numpy()
            best_x = tuple(best_x)
            best_y = Y[best_idx].numpy()
            if self.negate:
                best_y = -best_y
            return best_x, best_y

        else:
            raise NotImplementedError
