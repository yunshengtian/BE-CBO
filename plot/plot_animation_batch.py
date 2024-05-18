import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from plot_animation import plot_animation
from utils import parallel_execute


def plot_animation_parallel(log_dir, save_dir, n_seed, model_type, algo, algo_exclude, grid_size, n_level, prediction, num_proc):

    worker_args = []

    func_dirs = os.listdir(log_dir)

    for func_dir in func_dirs:
        log_path_func = os.path.join(log_dir, func_dir)
        if not os.path.isdir(log_path_func): continue

        model_type_dirs = os.listdir(log_path_func)
        for model_type_dir in model_type_dirs:
            if model_type is not None and model_type_dir not in model_type: continue

            log_path_model = os.path.join(log_path_func, model_type_dir)
            if not os.path.isdir(log_path_model): continue

            algo_dirs = os.listdir(log_path_model)

            for algo_dir in algo_dirs:
                if algo is not None and algo_dir not in algo: continue
                if algo_exclude is not None and algo_dir in algo_exclude: continue

                log_path_algo = os.path.join(log_path_model, algo_dir)
                if not os.path.isdir(log_path_algo): continue

                seeds = [f'{i}.pkl' for i in range(n_seed)]
                for seed in seeds:
                    log_path_seed = os.path.join(log_path_algo, seed)
                    if not log_path_seed.endswith('.pkl'): continue

                    os.makedirs(os.path.join(save_dir, func_dir, model_type_dir, algo_dir), exist_ok=True)
                    save_path_seed = os.path.join(save_dir, func_dir, model_type_dir, algo_dir, seed.replace('.pkl', ''))

                    worker_args.append([
                        log_path_seed, save_path_seed, False, grid_size, n_level, prediction,
                    ])

    try:
        for _ in parallel_execute(plot_animation, worker_args, [{} for _ in worker_args], num_proc):
            pass
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--model-type', type=str, nargs='+', default=None)
    parser.add_argument('--algo', type=str, nargs='+', default=None)
    parser.add_argument('--algo-exclude', type=str, nargs='+', default=None)
    parser.add_argument('--grid-size', type=int, default=100)
    parser.add_argument('--n-level', type=int, default=100)
    parser.add_argument('--num-proc', type=int, default=6)
    parser.add_argument('--n-seed', type=int, default=3)
    parser.add_argument('--prediction', default=False, action='store_true')
    args = parser.parse_args()

    plot_animation_parallel(args.log_dir, args.save_dir, args.n_seed, args.model_type, args.algo, args.algo_exclude, args.grid_size, args.n_level, args.prediction, args.num_proc)
