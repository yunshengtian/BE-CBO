
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from argparse import ArgumentParser
import torch

from plot.plot import plot_2D_fun, plot_2D_fun_single_fig
from test_functions import test_functions


parser = ArgumentParser()
parser.add_argument('--fun', type=str, required=True)
parser.add_argument('--n-level', type=int, default=100)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--no-negate', default=False, action='store_true')
parser.add_argument('--no-plot', default=False, action='store_true')
parser.add_argument('--single-fig', default=False, action='store_true')
parser.add_argument('--compute-optima', type=str, default=None)
parser.add_argument('--check-constr', default=False, action='store_true')
args = parser.parse_args()

fun = test_functions[args.fun](negate=not args.no_negate)

if not args.no_plot and fun.dim <= 2:
    if args.single_fig:
        plot_2D_fun_single_fig(fun, n_level=args.n_level, save_path=args.save_path)
    else:
        plot_2D_fun(fun, n_level=args.n_level, save_path=args.save_path)

if args.compute_optima is not None:
    best_x, best_y = fun.compute_optima(method=args.compute_optima)
else:
    best_x, best_y = fun._optimizers, fun._optimal_value
print('Best x:', best_x)
print('Best y:', best_y)

if args.check_constr:
    X = torch.atleast_2d(torch.Tensor(best_x))
    cons = fun.evaluate_slack(X=X, noise=False)
    print('Constr:', cons)
