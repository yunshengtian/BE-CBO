import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import pickle
import torch

from test_functions import test_functions
from plot import plot_progress_multi


def set_cfg(cfg, result):
    for key in cfg:
        if cfg[key] is None:
            cfg[key] = result[key]
        else:
            assert cfg[key] == result[key]


def plot_progress(log_paths, save_path, show):

    Y_list = []
    C_list = []
    legend_list = []

    cfg = {
        'fun_name': None,
        'n_init': None,
    }

    for log_path in log_paths:
        if not log_path.endswith('.pkl'): continue
        with open(log_path, 'rb') as fp:
            result = pickle.load(fp)

        set_cfg(cfg, result)

        Y_list.append(torch.from_numpy(result['Y']))
        C_list.append(torch.from_numpy(result['C']))
        legend_list.append(result['algo_name'])
    
    fun = test_functions[cfg['fun_name']](negate=True)
    title = f'{fun.__class__.__name__} (dim = {fun.dim})'
    plot_progress_multi(Y_list, C_list, legend_list=legend_list, optimal_value=fun.optimal_value, n_init=cfg['n_init'], title=title,
        save_path=save_path, show=show)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-path', type=str, nargs='+', required=True)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--show', default=False, action='store_true')
    args = parser.parse_args()

    plot_progress(args.log_path, args.save_path, args.show)
