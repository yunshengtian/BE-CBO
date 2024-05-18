import json
from argparse import ArgumentParser
import os
import re

parser = ArgumentParser()
parser.add_argument('--log-dir', type=str, required=True)
args = parser.parse_args()

spec_path = os.path.join(args.log_dir, 'spec.json')
if os.path.exists(spec_path):
    with open(spec_path, 'r') as fp:
        spec = json.load(fp)
    n_total = len(spec['fun']) * len(spec['algo']) * len(spec['reg_type']) * len(spec['cls_type']) * spec['n_seed']
else:
    n_total = None

n_completed = 0
for fun_name in os.listdir(args.log_dir):
    fun_dir = os.path.join(args.log_dir, fun_name)
    if os.path.isdir(fun_dir):
        for model_name in os.listdir(fun_dir):
            model_dir = os.path.join(fun_dir, model_name)
            if os.path.isdir(model_dir):
                for algo_name in os.listdir(model_dir):
                    algo_dir = os.path.join(model_dir, algo_name)
                    if os.path.isdir(algo_dir):
                        for file_name in os.listdir(algo_dir):
                            if re.match('^[0-9]+.pkl$', file_name) is not None:
                                n_completed += 1

if n_total is not None:
    print(f'Completed: {n_completed}/{n_total}')
else:
    print(f'Completed: {n_completed}')
