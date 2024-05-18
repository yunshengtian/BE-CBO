
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)
import json
import numpy as np

from utils import parallel_execute
from run_exp import run_exp
from run_exp_with_prediction import run_exp_with_prediction


def run_exp_parallel(fun_names, algo_names, reg_types, cls_types, n_seed, n_init, n_total,
    prediction, grid_size, n_level, log_dir, num_proc, device=None, model_save_iter=[], **kwargs):

    worker_args = []
    worker_kwargs = []
    for seed in range(n_seed):
        for fun_name in fun_names:
            for reg_type in reg_types:
                for cls_type in cls_types:
                    for algo_name in algo_names:
                        log_path = os.path.join(log_dir, fun_name, f'{reg_type}_{cls_type}', algo_name, str(seed))
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                        worker_args.append(
                            (fun_name, algo_name, reg_type, cls_type, seed, n_init, n_total, grid_size, n_level, log_path, False, False, device, model_save_iter)
                        )
                        worker_kwargs.append(kwargs)

    if prediction:
        worker = run_exp_with_prediction
    else:
        worker = run_exp
    try:
        for _ in parallel_execute(worker=worker, args=worker_args, kwargs=worker_kwargs, num_proc=num_proc):
            pass
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--fun', type=str, nargs='+', required=True)
    parser.add_argument('--algo', type=str, nargs='+', required=True)
    parser.add_argument('--reg-type', type=str, nargs='+', default=['gp'])
    parser.add_argument('--cls-type', type=str, nargs='+', default=['gp'])
    parser.add_argument('--n-seed', type=int, default=6)
    parser.add_argument('--n-init', type=int, default=10)
    parser.add_argument('--n-total', type=int, default=20)
    parser.add_argument('--prediction', default=False, action='store_true')
    parser.add_argument('--grid-size', type=int, default=100)
    parser.add_argument('--n-level', type=int, default=100)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--num-proc', type=int, default=6)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model-save-iter', type=int, nargs='+', default=[])
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    spec_path = os.path.join(args.log_dir, 'spec.json')
    with open(spec_path, 'w') as fp:
        json.dump(vars(args), fp)

    run_exp_parallel(args.fun, args.algo, args.reg_type, args.cls_type, args.n_seed, args.n_init, args.n_total, args.prediction, args.grid_size, args.n_level, args.log_dir, args.num_proc, 
        device=args.device, model_save_iter=args.model_save_iter)
