import os
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--log-dir', type=str, required=True)
parser.add_argument('--save-dir', type=str, required=True)
parser.add_argument('--fun', type=str, nargs='+', default=None)
parser.add_argument('--model-type', type=str, nargs='+')
parser.add_argument('--algo', type=str, nargs='+')
parser.add_argument('--n-total', type=int, default=None)
parser.add_argument('--n-seed', type=int, default=None)
parser.add_argument('--log-regret', default=False, action='store_true')
parser.add_argument('--names', type=str, nargs='+', default=None)
args = parser.parse_args()

try:
    model_type_str = f'--model-type {" ".join(args.model_type)}'
    algo_str = f'--algo {" ".join(args.algo)}'
    n_total_str = '' if args.n_total is None else f'--n-total {args.n_total}'
    n_seed_str = '' if args.n_seed is None else f'--n-seed {args.n_seed}'
    log_regret_str = '' if not args.log_regret else f'--log-regret'
    names_str = '' if args.names is None else f'--names {" ".join(args.names)}'

    if args.fun is None:
        funs = []
        for fun in os.listdir(args.log_dir):
            fun_dir = os.path.join(args.log_dir, fun)
            if os.path.isdir(fun_dir):
                funs.append(fun)
    else:
        funs = args.fun

    for fun in funs:
        fun_dir = os.path.join(args.log_dir, fun)

        curr_log_dir = f'{args.log_dir}/{fun}'
        curr_save_dir = f'{args.save_dir}/{fun}'

        cmd = f'python {project_base_dir}/plot/plot_progress_by_exp.py --log-dir {curr_log_dir} --save-dir {curr_save_dir} {model_type_str} {algo_str} {n_seed_str} {n_total_str} {log_regret_str} {names_str}'
        os.system(cmd)

except KeyboardInterrupt:
    pass
