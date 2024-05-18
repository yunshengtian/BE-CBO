import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import pickle
import torch

from test_functions import test_functions
from plot import plot_2D_performance, plot_2D_performance_with_prediction


def plot_animation(log_path, save_path, show, grid_size, n_level, prediction):
    if log_path.endswith('.pkl'): log_path = log_path[:-4]
    log_dir = os.path.dirname(log_path)
    log_basename = os.path.basename(log_path)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    result_grid = {}
    for file_name in os.listdir(log_dir):
        if file_name.endswith('.pkl') and file_name.startswith(log_basename) and 'grid' in file_name:
            log_grid_path = os.path.join(log_dir, file_name)
            len_X = int(file_name[:-4].split('_')[-1])
            with open(log_grid_path, 'rb') as fp:
                result_grid[len_X] = pickle.load(fp)
    result_grid = dict(sorted(result_grid.items()))

    if not os.path.exists(log_path + '.pkl'): return
    with open(log_path + '.pkl', 'rb') as fp:
        result = pickle.load(fp)

    fun_name = result['fun_name']
    fun = test_functions[fun_name](negate=True)

    if prediction:
        plot_2D_performance_with_prediction(fun, 
            torch.from_numpy(result['X']), log_path, result['n_init'], grid_size=grid_size, n_level=n_level,
            animation=True, save_path=save_path, show=show)
    else:
        plot_2D_performance(fun, 
            torch.from_numpy(result['X']), result['n_init'], grid_size=grid_size, n_level=n_level, 
            animation=True, save_path=save_path, show=show)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--show', default=False, action='store_true')
    parser.add_argument('--grid-size', type=int, default=100)
    parser.add_argument('--n-level', type=int, default=100)
    parser.add_argument('--prediction', default=False, action='store_true')
    args = parser.parse_args()

    plot_animation(args.log_path, args.save_path, args.show, args.grid_size, args.n_level, args.prediction)
