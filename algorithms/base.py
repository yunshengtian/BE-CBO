from abc import ABC, abstractmethod
from typing import Tuple, Union
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
import os

from model import fit_classifier, fit_regressor, predict_classifier, predict_regressor


class ConstrainedBO(ABC):

    def __init__(self, device: torch.device, reg_type='gp', cls_type='gp', eval_budget=None) -> None:
        """
        Initialize the optimization with fixed batch size and device.
        """
        self.device = device

        self.cached_model = None
        self.cached_constraint_models = None
        self.reg_type = reg_type
        self.cls_type = cls_type
        self.eval_budget = eval_budget

    def fit_surrogate(self, X: Tensor, Y: Tensor, C: Tensor, save_cache: bool = False, reg_kwargs: dict = {}, cls_kwargs: dict = {}) -> Tuple[SingleTaskGP, list]:
        """
        Fit surrogate objective and constraint models.

        Args:
            X: A `N x d`-dim tensor of existing designs.
            Y: A `N`-dim tensor of existing objective values.
            C: A `N x n_c`-dim tensor of existing constraint values.
            save_cache: If save surrogate objective and constraint models to cache to avoid refitting later.

        Returns:
            Fitted surrogate objective model.
            A list of fitted surrogate constraint models.
        """
        is_valid = (C >= 0.0).all(dim=-1)
        model = fit_regressor[self.reg_type](X[is_valid], Y[is_valid], model=self.cached_model, **reg_kwargs)
        C = (C >= 0.0).to(float)
        if self.cached_constraint_models is not None:
            constraint_models = [fit_classifier[self.cls_type](X, Ci, model_i, **cls_kwargs) for Ci, model_i in zip(C.T,self.cached_constraint_models)]
        else:
            constraint_models = [fit_classifier[self.cls_type](X, Ci, **cls_kwargs) for Ci in C.T]

        if save_cache:
            self.cached_model, self.cached_constraint_models = model, constraint_models
        return model, constraint_models

    @abstractmethod
    def optimize(self, X: Tensor, Y: Tensor, C: Tensor, load_cache: bool = False) -> Tensor:
        """
        Run 1 iteration of optimization, propose best points to evaluate.

        Args:
            X: A `N x d`-dim tensor of existing designs.
            Y: A `N`-dim tensor of existing objective values.
            C: A `N x n_c`-dim tensor of existing constraint values.
            load_cache: If load cached surrogate objective and constraint models to avoid refitting.

        Returns:
            A `1 x d`-dim tensor of new designs.
        """
        pass

    def predict(self, X: Tensor, Y: Tensor, C: Tensor, X_new: Tensor, load_cache: bool = False, mean_only: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Given existing data, predict the performance of new data.

        Args:
            X: A `N x d`-dim tensor of existing designs.
            Y: A `N`-dim tensor of existing objective values.
            C: A `N x n_c`-dim tensor of existing constraint values.
            X_new: A `M x d`-dim tensor of new designs.
            load_cache: If load cached surrogate objective and constraint models to avoid refitting.

        Returns:
            A `M`-dim tensor of predicted new objective values.
            A `M x n_c`-dim tensor of predicted new constraint values.
        """
        if load_cache:
            model, constraint_models = self.cached_model, self.cached_constraint_models
        else:
            model, constraint_models = self.fit_surrogate(X, Y, C)

        if mean_only:
            with torch.no_grad():
                Y_pred = predict_regressor(model, X_new, mean_only=mean_only)
                C_pred = []
                for constraint_model in constraint_models:
                    C_pred.append(predict_classifier(constraint_model, X_new, mean_only=mean_only))
            C_pred = torch.stack([C_pred_i for C_pred_i in C_pred], dim=-1)
            return Y_pred, C_pred

        else:
            with torch.no_grad():
                Y_pred_mean, Y_pred_var = predict_regressor(model, X_new, mean_only=mean_only)
                C_pred_mean, C_pred_var = [], []
                for constraint_model in constraint_models:
                    c_pred_mean, c_pred_var = predict_classifier(constraint_model, X_new, mean_only=mean_only)
                    C_pred_mean.append(c_pred_mean)
                    C_pred_var.append(c_pred_var)
            C_pred_mean = torch.stack([C_pred_i for C_pred_i in C_pred_mean], dim=-1)
            C_pred_var = torch.stack([C_pred_i for C_pred_i in C_pred_var], dim=-1)
            return Y_pred_mean, Y_pred_var, C_pred_mean, C_pred_var

    def acquisition(self, X: Tensor, Y: Tensor, C: Tensor, X_new: Tensor, load_cache: bool = False) -> Tensor:
        """
        Given existing data, output the acquisition of new data.

        Args:
            X: A `N x d`-dim tensor of existing designs.
            Y: A `N`-dim tensor of existing objective values.
            C: A `N x n_c`-dim tensor of existing constraint values.
            X_new: A `M x d`-dim tensor of new designs.
            load_cache: If load cached surrogate objective and constraint models to avoid refitting.

        Returns:
            A `M`-dim tensor of new acquisition values.
        """
        raise NotImplementedError

    def save_model(self, log_path, save_iter):

        assert self.cached_model is not None and self.cached_constraint_models is not None
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        model_path = f'{log_path}_model_{save_iter}.pth'
        torch.save(self.cached_model.get_state_dict(), model_path)
        
        for i, constraint_model in enumerate(self.cached_constraint_models):
            constraint_model_path = f'{log_path}_constr_model_{i}_{save_iter}.pth'
            torch.save(constraint_model.get_state_dict(), constraint_model_path)
