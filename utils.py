from typing import Tuple
import warnings
import numpy as np
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
import multiprocessing as mp
from tqdm import tqdm
import traceback
import pickle


def set_environment(seed: int = 0) -> torch.device:
    """
    Initial environment setup.
    """
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.double)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return device


def get_initial_points(dim: int, n_pts: int) -> Tensor:
    """
    Generate initial design points.
    """
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts)
    return X_init


def get_initial_dataset(fun, n_pts: int, device: str) -> Tensor:
    """
    Generate initial design points.
    """
    has_valid = False
    while not has_valid:
        sobol = SobolEngine(dimension=fun.dim, scramble=True)
        X_init = sobol.draw(n=n_pts).to(device)
        Y_init, C_init = fun(X_init)
        has_valid = (C_init >= 0).all(dim=-1).any()
    return X_init, Y_init, C_init


def get_best_value(X: Tensor, Y: Tensor, C: Tensor) -> Tuple[Tensor, float]:
    """
    Find the best objective value.
    """
    assert Y.dim() == 1, 'only support single objective'
    is_valid = (C >= 0.0).all(dim=-1)
    if is_valid.any():
        best_idx = Y[is_valid].argmax()
        best_x, best_y = X[is_valid][best_idx], Y[is_valid][best_idx].item()
        return best_x, best_y
    else:
        return None, float("-inf")


def normalize(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return (X - bounds[0]) / (bounds[1] - bounds[0])


def unnormalize(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return X * (bounds[1] - bounds[0]) + bounds[0]


def parallel_worker(worker, args, kwargs, queue, proc_idx):
    result = worker(*args, **kwargs)
    queue.put([result, proc_idx])


def parallel_execute(worker, args, kwargs, num_proc, show_progress=True, desc=None, terminate_func=None, raise_exception=True):
    '''
    Tool for parallel execution
    '''
    if show_progress:
        pbar = tqdm(total=len(args), desc=desc)

    queue = mp.Queue()
    procs = {}
    n_active_proc = 0

    try:

        # loop over arguments for all processes
        for proc_idx, (arg, kwarg) in enumerate(zip(args,kwargs)):

            if num_proc > 1:
                proc = mp.Process(target=parallel_worker, args=(worker, arg, kwarg, queue, proc_idx))
                proc.start()
                procs[proc_idx] = proc
                n_active_proc += 1

                if n_active_proc >= num_proc: # launch a new process after an existing one finishes
                    result, proc_idx = queue.get()
                    procs.pop(proc_idx)
                    yield result

                    if terminate_func and terminate_func(result): # terminate condition meets
                        for p in procs.values(): # terminate all running processes
                            p.terminate()
                        if show_progress:
                            pbar.update(pbar.total - pbar.last_print_n)
                            pbar.close()
                        return
                    
                    n_active_proc -= 1

                    if show_progress:
                        pbar.update(1)
            else:
                parallel_worker(worker, arg, kwarg, queue, proc_idx) # no need to use mp.Process when serial
                result, _ = queue.get()
                yield result

                if terminate_func and terminate_func(result): # terminate condition meets
                    if show_progress:
                        pbar.update(pbar.total - pbar.last_print_n)
                        pbar.close()
                    return

                if show_progress:
                    pbar.update(1)

        for _ in range(n_active_proc): # wait for existing processes to finish
            result, proc_idx = queue.get()
            procs.pop(proc_idx)
            yield result

            if terminate_func and terminate_func(result): # terminate condition meets
                for p in procs.values(): # terminate all running processes
                    p.terminate()
                if show_progress:
                    pbar.update(pbar.total - pbar.last_print_n)
                    pbar.close()
                return

            if show_progress:
                pbar.update(1)

    except (Exception, KeyboardInterrupt) as e:
        if type(e) == KeyboardInterrupt:
            print('[parallel_execute] interrupt')
        else:
            print('[parallel_execute] exception:', e)
            print(traceback.format_exc())
        for proc in procs.values():
            proc.terminate()
        if raise_exception:
            raise e

    if show_progress:
        pbar.close()
