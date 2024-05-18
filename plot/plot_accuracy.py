import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from test_functions import test_functions
from utils import normalize, unnormalize


def compute_accuracy(log_path, grid_size):
    if log_path.endswith('.pkl'): log_path = log_path[:-4]
    log_dir = os.path.dirname(log_path)
    log_basename = os.path.basename(log_path)

    with open(log_path + '.pkl', 'rb') as fp:
        result = pickle.load(fp)

    result_grid = {}
    for file_name in os.listdir(log_dir):
        if file_name.endswith('.pkl') and file_name.startswith(log_basename) and 'grid' in file_name:
            log_grid_path = os.path.join(log_dir, file_name)
            len_X = int(file_name[:-4].split('_')[-1])
            with open(log_grid_path, 'rb') as fp:
                result_grid[len_X] = pickle.load(fp)
    result_grid = dict(sorted(result_grid.items()))
    result['C_grid_list'] = [x['C'] for x in result_grid.values()]

    fun_name = result['fun_name']
    fun = test_functions[fun_name](negate=True)

    assert fun.dim == 2
    bounds = fun.bounds.numpy()

    X0_grid, X1_grid = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    X_vec = np.hstack((X0_grid.reshape(-1, 1), X1_grid.reshape(-1, 1)))
    Y_vec, C_vec = fun(torch.from_numpy(X_vec))
    Y_vec, C_vec = Y_vec.numpy(), C_vec.numpy()
    valid_vec = (C_vec >= 0).all(axis=1)
    valid_grid = valid_vec.reshape(grid_size, grid_size).astype(float)

    n_init = result['n_init']
    # evaluated designs
    X = result['X']
    X = unnormalize(X, bounds)
    n_batch = len(X) - n_init

    accuracy_his = []
    C_grid_list =  result['C_grid_list']
    for i in range(n_batch):
        acc = np.isclose(np.round(C_grid_list[i]), valid_grid).sum() / (grid_size * grid_size)
        accuracy_his.append(acc)

    return accuracy_his


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-path', type=str, required=True)
    parser.add_argument('--grid-size', type=int, default=100)
    args = parser.parse_args()

    acc_his = compute_accuracy(args.log_path, args.grid_size)
    print(acc_his)