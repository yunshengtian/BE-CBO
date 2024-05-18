import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import matplotlib.pyplot as plt
import numpy as np

from plot_accuracy import compute_accuracy
from utils import parallel_execute


def plot_accuracy_parallel(log_dir, save_dir, algo, n_seed, grid_size):

    for func_dir in os.listdir(log_dir):
        log_path_func = os.path.join(log_dir, func_dir)
        if not os.path.isdir(log_path_func): continue

        os.makedirs(save_dir, exist_ok=True)
        save_path_func = os.path.join(save_dir, f'{func_dir}.png')

        fig, _ = plt.subplots(1, 1)

        algo_dirs = os.listdir(log_path_func)
        algo_dirs.sort()
        algo_dirs = [dir for dir in algo_dirs if not dir.startswith('bandit')] + [dir for dir in algo_dirs if dir.startswith('bandit')]

        for algo_dir in algo_dirs:
            if algo is not None and algo_dir.split('_')[0] not in algo: continue
            log_path_algo = os.path.join(log_path_func, algo_dir)
            if not os.path.isdir(log_path_algo): continue

            seeds = [f'{i}.pkl' for i in range(n_seed)]
            acc_his_all = []
            for seed in seeds:
                log_path_seed = os.path.join(log_path_algo, seed)
                if not log_path_seed.endswith('.pkl'): continue

                acc_his = compute_accuracy(log_path_seed, grid_size)
                acc_his_all.append(acc_his)

            acc_his_mean, acc_his_std = np.mean(acc_his_all, axis=0), np.std(acc_his_all, axis=0)
            plt.plot(range(len(acc_his_mean)), acc_his_mean, label=algo_dir.split('_')[0])
            plt.fill_between(range(len(acc_his_std)), acc_his_mean - 0.5 * acc_his_std, acc_his_mean + 0.5 * acc_his_std, alpha=0.2)

        plt.title(f'Problem: {func_dir}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path_func)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--algo', nargs='+', type=str, default=None)
    parser.add_argument('--grid-size', type=int, default=100)
    parser.add_argument('--n-seed', type=int, default=1)
    args = parser.parse_args()

    plot_accuracy_parallel(args.log_dir, args.save_dir, args.algo, args.n_seed, args.grid_size)
