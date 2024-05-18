
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import pickle
import torch
from botorch.utils.transforms import unnormalize

from algorithms import algorithms
from test_functions import test_functions
from utils import set_environment, get_initial_dataset, get_best_value
from plot.plot import plot_progress_single, plot_2D_performance_with_prediction


def run_exp_with_prediction(fun_name, algo_name, reg_type, cls_type, seed, n_init, n_total, grid_size, n_level, log_path, plot, verbose=True, device = None, model_save_iter=[], **kwargs):

    print(f'Experiment started: {fun_name} {algo_name} {reg_type} {cls_type} {seed}')

    # Set environment
    if device is None:
        device = set_environment(seed=seed)
    else:
        set_environment(seed=seed)

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Define the test function 
    fun = test_functions[fun_name](negate=True).to(device)
    fun.short_name = fun_name

    # Build algorithm
    algo = algorithms[algo_name](device=device, reg_type=reg_type, cls_type=cls_type, eval_budget=n_total)
    algo.fun = fun
    algo.seed = seed

    # Get initial data
    X, Y, C = get_initial_dataset(fun, n_pts=n_init, device=device) # Initial designs, objective and constraint values

    X0_grid, X1_grid = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    X_new = np.hstack((X0_grid.reshape(-1, 1), X1_grid.reshape(-1, 1)))
    X_new = torch.from_numpy(X_new).to(device)

    # Main optimization loop
    while len(X) < n_total:

        # Generate a batch of candidates
        algo.fit_surrogate(X, Y, C, save_cache=True)
        X_next = algo.optimize(X, Y, C, load_cache=True)

        if len(X) in model_save_iter:
            algo.save_model(log_path, len(X))

        # Evaluate both the objective and constraints for the selected candidates 
        Y_next, C_next = fun(X_next)

        Y_new_mean, Y_new_var, C_new_mean, C_new_var = algo.predict(X, Y, C, X_new, load_cache=True, mean_only=False)
        C_new_mean = C_new_mean.prod(dim=-1)
        C_new_var = C_new_var.prod(dim=-1)
        Y_grid_mean = torch.reshape(Y_new_mean, (grid_size, grid_size))
        Y_grid_var = torch.reshape(Y_new_var, (grid_size, grid_size))
        C_grid_mean = torch.reshape(C_new_mean, (grid_size, grid_size))
        C_grid_var = torch.reshape(C_new_var, (grid_size, grid_size))
        grid_data = {
            'Y_mean': Y_grid_mean.detach().to('cpu').numpy(), 
            'Y_var': Y_grid_var.detach().to('cpu').numpy(), 
            'C_mean': C_grid_mean.detach().to('cpu').numpy(),
            'C_var': C_grid_var.detach().to('cpu').numpy(),
        }

        try:
            with torch.no_grad():
                A_new = algo.acquisition(X, Y, C, X_new, load_cache=True)
            A_grid = torch.reshape(A_new, (grid_size, grid_size))
            # A_grid_list.append(A_grid.detach().cpu().numpy())
            grid_data['A'] = A_grid.detach().to('cpu').numpy()
        except NotImplementedError:
            pass

        if log_path is not None:
            with open(log_path + f'_grid_{len(X)}' + '.pkl', 'wb') as fp:
                pickle.dump(grid_data, fp)

        # Append data
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
        C = torch.cat((C, C_next), dim=0)

        # Print the best value found so far
        best_x, best_y = get_best_value(X, Y, C)
        best_x = unnormalize(best_x, fun.bounds)
        if verbose:
            print(
                f"{len(X)}) Best value: {best_y:.2e} | Best point: {best_x.round(decimals=3).tolist()}"
            )

    # Log result
    if log_path is not None:
        result = {
            'fun_name': fun_name, 'algo_name': algo_name, 'seed': seed,
            'n_init': n_init, 'n_total': n_total, 'grid_size': grid_size,
            'X': X.numpy(), 'Y': Y.numpy(), 'C': C.numpy(),
        }
        with open(log_path + '.pkl', 'wb') as fp:
            pickle.dump(result, fp)

    # Plot optimization progress
    if plot:
        title = f'{fun.__class__.__name__}\n ({fun.dim}D)'
        plot_progress_single(Y, C, legend=algo_name, optimal_value=fun.optimal_value, n_init=n_init, title=title)
        if log_path is not None:
            plot_2D_performance_with_prediction(fun, X, log_path, n_init, grid_size=grid_size, n_level=n_level, animation=True, save_path=log_path)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--fun', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--reg-type', type=str, default='gp')
    parser.add_argument('--cls-type', type=str, default='gp')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-init', type=int, default=10)
    parser.add_argument('--n-total', type=int, default=20)
    parser.add_argument('--grid-size', type=int, default=100)
    parser.add_argument('--n-level', type=int, default=100)
    parser.add_argument('--log-path', type=str, default=None)
    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model-save-iter', type=int, nargs='+', default=[])
    args = parser.parse_args()

    run_exp_with_prediction(args.fun, args.algo, args.reg_type, args.cls_type, args.seed, args.n_init, args.n_total, args.grid_size, args.n_level, args.log_path, args.plot, device=args.device, model_save_iter=args.model_save_iter)
