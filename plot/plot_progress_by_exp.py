import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import pickle
import torch

from test_functions import test_functions
from plot import plot_progress_all


def set_cfg(cfg, result):
    for key in cfg:
        if cfg[key] is None:
            cfg[key] = result[key]
        else:
            assert cfg[key] == result[key]


def plot_progress(log_dir, save_dir, model_types, algos, n_total=None, n_seed=None, show=False, log_regret=False, names=None):

    Y_all = []
    C_all = []
    legend_list = []

    cfg = {
        'fun_name': None,
        'n_init': None,
    }

    if names is None: names = [None] * len(model_types)
    assert len(names) == len(model_types) == len(algos)

    for model_type, algo, name in zip(model_types, algos, names):

        log_model_dir = os.path.join(log_dir, model_type)
        assert os.path.isdir(log_model_dir)

        log_algo_dir = os.path.join(log_model_dir, algo)
        if not os.path.isdir(log_algo_dir): continue

        Y_algo = []
        C_algo = []

        seed_dirs = sorted(os.listdir(log_algo_dir))
        for seed in seed_dirs:
            if len(seed) > 5: continue
            log_seed_path = os.path.join(log_algo_dir, seed)
            if not log_seed_path.endswith('.pkl'): continue

            if n_seed is not None and int(seed.replace('.pkl', '')) >= n_seed: continue

            with open(log_seed_path, 'rb') as fp:
                result = pickle.load(fp)

            set_cfg(cfg, result)

            Y = torch.from_numpy(result['Y'])
            C = torch.from_numpy(result['C'])
            if n_total is not None:
                Y = Y[:n_total]
                C = C[:n_total]

            Y_algo.append(Y)
            C_algo.append(C)
        
        try:
            Y_all.append(-torch.stack(Y_algo))
            C_all.append(torch.stack(C_algo))
        except:
            print(f'No algo {algo} in {log_dir}, skipped')
            continue

        exp = f'{algo} ({model_type})'

        if name is None:
            legend_list.append(exp)
        else:
            legend_list.append(name)

    if cfg['fun_name'] is None:
        print(f'No experiments in {log_dir} with model {model_type}, skipped')
        return

    fun = test_functions[cfg['fun_name']](negate=False)
    title = fun.name + f'\n ({fun.dim}D)'
    
    subtitle = None
    save_path = os.path.join(save_dir, 'performance')
    plot_progress_all(Y_all, C_all, legend_list, fun.optimal_value, cfg['n_init'], title, subtitle, save_path, show, log_regret)

    
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True, help='log_dir/func/')
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--model-type', type=str, nargs='+', required=True)
    parser.add_argument('--algo', type=str, nargs='+', required=True)
    parser.add_argument('--n-total', type=int, default=None)
    parser.add_argument('--n-seed', type=int, default=None)
    parser.add_argument('--show', default=False, action='store_true')
    parser.add_argument('--log-regret', default=False, action='store_true')
    parser.add_argument('--names', type=str, nargs='+', default=None)
    args = parser.parse_args()

    plot_progress(args.log_dir, args.save_dir, args.model_type, args.algo, args.n_total, args.n_seed, args.show, args.log_regret, args.names)
