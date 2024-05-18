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
from plot.plot import plot_progress_single, plot_2D_performance

from time import time
import datetime


def run_exp(fun_name, algo_name, reg_type, cls_type, seed, n_init, n_total, grid_size, n_level, log_path, plot, verbose=True, device = None, model_save_iter=[], **kwargs):

    print(f'Experiment started: {fun_name} {algo_name} {reg_type} {cls_type} {seed} ({datetime.datetime.now()})')

    # Set environment
    if device is None:
        device = set_environment(seed=seed)
    else:
        set_environment(seed=seed)

    # Define the test function 
    fun = test_functions[fun_name](negate=True).to(device)
    fun.short_name = fun_name

    # Build algorithm
    algo = algorithms[algo_name](device=device, reg_type=reg_type, cls_type=cls_type, eval_budget=n_total)
    algo.fun = fun
    algo.seed = seed

    # Get initial data
    X, Y, C = get_initial_dataset(fun, n_pts=n_init, device=device) # Initial designs, objective and constraint values

    t_start = time()
    t_all = []

    # Main optimization loop
    while len(X) < n_total:

        # Generate a batch of candidates
        if algo_name.startswith('random'):
            X_next = algo.optimize(X, Y, C)
        else:
            algo.fit_surrogate(X, Y, C, save_cache=True)
            X_next = algo.optimize(X, Y, C, load_cache=True)

        if len(X) in model_save_iter:
            algo.save_model(log_path, len(X))

        # Evaluate both the objective and constraints for the selected candidates 
        Y_next, C_next = fun(X_next)

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

        t_all.append(time() - t_start)

    # Log result
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        result = {
            'fun_name': fun_name, 'algo_name': algo_name, 'seed': seed,
            'n_init': n_init, 'n_total': n_total, 
            'X': X.detach().cpu().numpy(), 'Y': Y.detach().cpu().numpy(), 'C': C.detach().cpu().numpy(),
            't': t_all,
        }
        with open(log_path + '.pkl', 'wb') as fp:
            pickle.dump(result, fp)

    # Plot optimization progress
    if plot:
        title = f'{fun.__class__.__name__}\n ({fun.dim}D)'
        plot_progress_single(Y, C, legend=algo_name, optimal_value=fun.optimal_value, n_init=n_init, title=title)
        plot_2D_performance(fun, X, n_init, grid_size=grid_size, n_level=n_level, animation=True, save_path=log_path)


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

    run_exp(args.fun, args.algo, args.reg_type, args.cls_type, args.seed, args.n_init, args.n_total, args.grid_size, args.n_level, args.log_path, args.plot, device=args.device, model_save_iter=args.model_save_iter)
