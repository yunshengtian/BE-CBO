import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from typing import List
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import rc, ticker
import pickle

from test_functions.base import ConstrainedSyntheticTestFunction
from utils import normalize, unnormalize


def _fill_invalid_values(x, fx, n_init, n_total):

    assert (x <= n_init).any()

    new_fx = [np.inf]
    curr_x = n_init + 1

    for i in range(len(x)):

        if x[i] <= n_init: # set initial value
            new_fx[0] = min(new_fx[0], fx[i])
            continue

        while curr_x < x[i]: # fill invalid values
            new_fx.append(new_fx[-1])
            curr_x += 1
        
        # fill current value
        assert curr_x == x[i]
        assert new_fx[-1] >= fx[i]
        new_fx.append(fx[i])
        curr_x += 1

    while curr_x <= n_total:
        new_fx.append(new_fx[-1])
        curr_x += 1

    return new_fx


def plot_progress_all(Y_all: Tensor, C_all: Tensor, legend_list: List[str], 
    optimal_value: float = None, n_init: int = 1, title: str = None, subtitle: str = None, save_path: str = None, show: bool = True,
    log_regret: bool = False) -> None:
    """
    Plot progress for multiple experiments with different algos and seeds.

    Args:
        Y_all: A `n_algos x n_seeds x (n_total - n_init)`-dim tensor.
        C_all: A `n_algos x n_seeds x (n_total - n_init) x nC`-dim tensor.
    """
    if log_regret: assert optimal_value is not None

    fig, ax = plt.subplots(figsize=(8, 8))
    min_y, max_y = float("inf"), float("-inf")

    # see https://matplotlib.org/stable/gallery/color/named_colors.html
    colors = ['slategray', 'orange', 'mediumaquamarine', 'dodgerblue', \
        'tomato', 'mediumslateblue', 'sienna', 'darkviolet', 'darkgreen', \
        'teal', 'crimson']

    assert len(colors) >= len(legend_list), 'colors not enough'

    for idx, (Y_algo, C_algo, legend) in enumerate(zip(Y_all, C_all, legend_list)): # algo level data

        fx_algo = []
        n_total = None
        
        for Y_seed, C_seed in zip(Y_algo, C_algo): # seed level data

            assert Y_seed.dim() == 1, 'multi-objective is not supported yet'

            # filter valid data
            is_valid = (C_seed >= 0.0).all(dim=-1).numpy()
            x = np.arange(1, len(C_seed) + 1)[is_valid]
            fx = np.minimum.accumulate(Y_seed[is_valid].cpu().numpy())

            n_total = len(Y_seed)
            fx_complete = _fill_invalid_values(x, fx, n_init, n_total)

            if log_regret:
                fx_complete = [np.log(optimal_value - val) for val in fx_complete]

            fx_algo.append(fx_complete)

        fx_algo = np.vstack(fx_algo)
        x_algo = np.arange(n_init, n_init + fx_algo.shape[1])

        if n_total is None:
            n_total = n_init + fx_algo.shape[1]
        else:
            n_total = max(n_total, n_init + fx_algo.shape[1])

        fx_mean, fx_std = fx_algo.mean(axis=0), fx_algo.std(axis=0)
        fx_lower, fx_upper = fx_mean - 0.5 * fx_std, fx_mean + 0.5 * fx_std
        plt.plot(x_algo, fx_mean, marker="", lw=3, label=legend, color=colors[idx])
        plt.fill_between(x_algo, fx_lower, fx_upper, color=colors[idx], alpha=0.1)

        min_y = min(min_y, np.min(fx_lower))
        max_y = max(max_y, np.max(fx_upper))

    if optimal_value is not None and not log_regret:
        plt.plot([n_init, n_total], [optimal_value, optimal_value], "k--", lw=3, label='Global optimum')
        min_y = min(optimal_value, min_y)
        max_y = max(optimal_value, max_y)

    extent_y = max_y - min_y

    plt.xlabel("Function value", fontsize=16)
    plt.xlabel("Number of evaluations", fontsize=16)
    if title is not None:
        plt.suptitle(title, fontsize=20)
    if subtitle is not None:
        plt.title(subtitle, fontsize=16)
    plt.xlim([n_init, n_total])
    plt.ylim([min_y - 0.05 * extent_y, max_y + 0.05 * extent_y])

    plt.grid(True)
    plt.tight_layout()
    plt.legend(
        fontsize=16,
    )
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + '.png')
    if show:
        plt.show()


def plot_progress_multi(Y_list: List[Tensor], C_list: List[Tensor], legend_list: List[str], 
    optimal_value: float = None, n_init: int = 1, title: str = None, save_path: str = None, show: bool = True) -> None:
    """
    Plot progress for multiple experiments.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    min_y, max_y = float("inf"), float("-inf")

    for Y, C, legend in zip(Y_list, C_list, legend_list):

        assert Y.dim() == 1, 'multi-objective is not supported yet'

        # filter valid data
        is_valid = (C >= 0.0).all(dim=-1).numpy()
        x = np.arange(1, len(C) + 1)[is_valid]
        fx = np.maximum.accumulate(Y[is_valid].cpu())

        plot_indices = x >= n_init

        # calculate start points
        if (x <= n_init).any():
            x_start, fx_start = n_init, fx[x <= n_init][-1]
            x = np.append(x_start, x[plot_indices])
            fx = np.append(fx_start, fx[plot_indices])

        # calculate end points
        x_end, fx_end = len(C), fx[-1]
        x = np.append(x, x_end)
        fx = np.append(fx, fx_end)

        plt.plot(x, fx, marker="", lw=3, label=legend)

        min_y = min(fx[0], min_y)
        max_y = max(fx[-1], max_y)

    num_data = np.max([len(Y) for Y in Y_list])

    if optimal_value is not None:
        plt.plot([n_init, num_data], [optimal_value, optimal_value], "k--", lw=3, label='Global optimum')
        min_y = min(optimal_value, min_y)
        max_y = max(optimal_value, max_y)

    extent_y = max_y - min_y

    plt.xlabel("Function value", fontsize=16)
    plt.xlabel("Number of evaluations", fontsize=16)
    if title is not None:
        plt.title(title, fontsize=20)
    plt.xlim([n_init, num_data])
    plt.ylim([min_y - 0.05 * extent_y, max_y + 0.05 * extent_y])

    plt.grid(True)
    plt.tight_layout()
    plt.legend(
        fontsize=16,
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + '.png')
    if show:
        plt.show()


def plot_progress_single(Y: Tensor, C: Tensor, legend: str, optimal_value: float = None, n_init: int = 1, title: str = None, save_path: str = None, show: bool = True) -> None:
    """
    Plot progress for a single experiment.
    """
    plot_progress_multi([Y], [C], [legend], optimal_value=optimal_value, n_init=n_init, title=title, save_path=save_path, show=show)


def plot_2D_fun(fun: ConstrainedSyntheticTestFunction, grid_size: int = 100, n_level: int = 10, save_path: str = None, show: bool = True) -> None:
    """
    Plot function landscape over a 2D design space.
    """
    assert fun.dim == 2
    bounds = fun.bounds.numpy()
    locator = ticker.MaxNLocator(n_level)

    X0_grid, X1_grid = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    X_vec = np.hstack((X0_grid.reshape(-1, 1), X1_grid.reshape(-1, 1)))
    Y_vec, C_vec = fun(torch.from_numpy(X_vec))
    Y_vec, C_vec = Y_vec.numpy(), C_vec.numpy()
    valid_vec = (C_vec >= 0).all(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    Y_grid = Y_vec.reshape(grid_size, grid_size)
    X_vec = unnormalize(X_vec, bounds)
    X0_grid, X1_grid = X_vec[:, 0].reshape(grid_size, grid_size), X_vec[:, 1].reshape(grid_size, grid_size)
    valid_grid = valid_vec.reshape(grid_size, grid_size).astype(float)
    Y_valid_grid = (Y_grid - Y_grid.min()) * valid_grid + Y_grid.min()

    # objective function plot
    pcolormesh = axes[0].contourf(X0_grid, X1_grid, Y_grid, cmap='Blues', locator=locator)
    fig.colorbar(pcolormesh, ax=axes[0])
    axes[0].set_title('Objective')

    # constraint function plot
    pcolormesh = axes[1].contourf(X0_grid, X1_grid, valid_grid, cmap='Blues', locator=locator)
    axes[1].contour(X0_grid, X1_grid, valid_grid, [0.5])
    fig.colorbar(pcolormesh, ax=axes[1])
    axes[1].set_title('Constraint')

    # objective + constraint function plot
    pcolormesh = axes[2].contourf(X0_grid, X1_grid, Y_valid_grid, cmap='Blues', locator=locator)
    fig.colorbar(pcolormesh, ax=axes[2])
    axes[2].set_title('Constrained Objective')

    if fun._optimizers != None:
        optimizers = np.array(fun._optimizers)
    else:
        optimizers = np.array([fun.compute_optima()[0]])
    axes[2].scatter(optimizers[:, 0], optimizers[:, 1], marker='*', c='red', s=100)

    for i in range(3):
        axes[i].axis([X0_grid.min(), X0_grid.max(), X1_grid.min(), X1_grid.max()])
        axes[i].set_xlabel('x1')
        axes[i].set_ylabel('x2')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    def onclick(event):
        x_click = np.array([event.xdata, event.ydata])
        x_click = normalize(x_click, bounds)[None, :]
        y_click, c_click = fun(torch.from_numpy(x_click))
        y_click, c_click = y_click.numpy()[0], c_click.numpy()[0]
        print(f'Objective: {y_click}, Constraint: {c_click >= 0.0}')

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + '.png')
    if show:
        plt.show()


def plot_2D_fun_single_fig(fun: ConstrainedSyntheticTestFunction, grid_size: int = 100, n_level: int = 10, save_path: str = None, show: bool = True) -> None:
    """
    Plot function landscape over a 2D design space.
    """
    assert fun.dim == 2
    bounds = fun.bounds.numpy()
    locator = ticker.MaxNLocator(n_level)

    X0_grid, X1_grid = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    X_vec = np.hstack((X0_grid.reshape(-1, 1), X1_grid.reshape(-1, 1)))
    Y_vec, C_vec = fun(torch.from_numpy(X_vec))
    Y_vec, C_vec = Y_vec.numpy(), C_vec.numpy()
    valid_vec = (C_vec >= 0).all(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    Y_grid = Y_vec.reshape(grid_size, grid_size)
    X_vec = unnormalize(X_vec, bounds)
    X0_grid, X1_grid = X_vec[:, 0].reshape(grid_size, grid_size), X_vec[:, 1].reshape(grid_size, grid_size)
    valid_grid = valid_vec.reshape(grid_size, grid_size).astype(float)
    Y_valid_grid = (Y_grid - Y_grid.min()) * valid_grid + Y_grid.min()

    # objective + constraint function plot
    pcolormesh = ax.contourf(X0_grid, X1_grid, Y_valid_grid, cmap='Blues', locator=locator)
    fig.colorbar(pcolormesh, ax=ax)
    ax.set_title(fun.name, fontsize=16)

    if fun._optimizers != None:
        optimizers = np.array(fun._optimizers)
    else:
        optimizers = np.array([fun.compute_optima()[0]])
    ax.scatter(optimizers[:, 0], optimizers[:, 1], marker='*', c='red', s=100)

    ax.axis([X0_grid.min(), X0_grid.max(), X1_grid.min(), X1_grid.max()])
    ax.set_xlabel('x1', fontsize=16)
    ax.set_ylabel('x2', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

    def onclick(event):
        x_click = np.array([event.xdata, event.ydata])
        x_click = normalize(x_click, bounds)[None, :]
        y_click, c_click = fun(torch.from_numpy(x_click))
        y_click, c_click = y_click.numpy()[0], c_click.numpy()[0]
        print(f'Objective: {y_click}, Constraint: {c_click >= 0.0}')

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_2D_performance(fun: ConstrainedSyntheticTestFunction, X: Tensor, n_init: int, grid_size: int = 100, n_level: int = 10, 
    animation: bool = False, save_path: str = None, show: bool = True) -> None:
    """
    Plot performance over a 2D design space.
    """
    assert fun.dim == 2
    bounds = fun.bounds.numpy()
    locator = ticker.MaxNLocator(n_level)

    X0_grid, X1_grid = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    X_vec = np.hstack((X0_grid.reshape(-1, 1), X1_grid.reshape(-1, 1)))
    Y_vec, C_vec = fun(torch.from_numpy(X_vec))
    Y_vec, C_vec = Y_vec.numpy(), C_vec.numpy()
    valid_vec = (C_vec >= 0).all(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    Y_grid = Y_vec.reshape(grid_size, grid_size)
    X_vec = unnormalize(X_vec, bounds)
    X0_grid, X1_grid = X_vec[:, 0].reshape(grid_size, grid_size), X_vec[:, 1].reshape(grid_size, grid_size)
    valid_grid = valid_vec.reshape(grid_size, grid_size).astype(float)
    Y_valid_grid = (Y_grid - Y_grid.min()) * valid_grid + Y_grid.min()

    # objective function plot
    pcolormesh = axes[0].contourf(X0_grid, X1_grid, Y_grid, cmap='Blues', locator=locator)
    fig.colorbar(pcolormesh, ax=axes[0])
    axes[0].set_title('Objective')

    # constraint function plot
    pcolormesh = axes[1].contourf(X0_grid, X1_grid, valid_grid, cmap='Blues', locator=locator)
    axes[1].contour(X0_grid, X1_grid, valid_grid, [0.5])
    fig.colorbar(pcolormesh, ax=axes[1])
    axes[1].set_title('Constraint')

    # objective + constraint function plot
    pcolormesh = axes[2].contourf(X0_grid, X1_grid, Y_valid_grid, cmap='Blues', locator=locator)
    fig.colorbar(pcolormesh, ax=axes[2])
    axes[2].set_title('Constrained Objective')

    # evaluated designs
    X = unnormalize(X.numpy(), bounds)
    n_batch = len(X) - n_init
    alphas = [0.1] * n_init
    for i, alpha in zip(range(n_batch), np.linspace(0.1, 1.0, n_batch + 1)[1:]):
        alphas.append(alpha)
    alphas = alphas[:len(X)]

    if fun._optimizers != None:
        optimizers = np.array(fun._optimizers)
        axes[2].scatter(optimizers[:, 0], optimizers[:, 1], marker='*', c='red', s=100)

    for i in range(3):
        axes[i].axis([X0_grid.min(), X0_grid.max(), X1_grid.min(), X1_grid.max()])
        axes[i].set_xlabel('x1')
        axes[i].set_ylabel('x2')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    def onclick(event):
        x_click = np.array([event.xdata, event.ydata])
        x_click = normalize(x_click, bounds)[None, :]
        y_click, c_click = fun(torch.from_numpy(x_click))
        y_click, c_click = y_click.numpy()[0], c_click.numpy()[0]
        print(f'Objective: {y_click}, Constraint: {c_click >= 0.0}')

    fig.canvas.mpl_connect('button_press_event', onclick)

    if not animation:
        for i in range(3):
            axes[i].scatter(X[:, 0], X[:, 1], c='black', alpha=alphas[-1])
    else:
        scatters = []
        for i in range(3):
            scatters.append(axes[i].scatter([], [], c='black'))

        def update(frame_number):
            len_X_frame = min(frame_number + n_init, len(X))
            for i in range(3):
                scatters[i].set_offsets(X[:len_X_frame])
                scatters[i].set_alpha(alphas[len_X_frame-1])
            return scatters

        anim = FuncAnimation(fig, update, frames=n_batch + 1, interval=300)
        if save_path is not None:
            anim.save(save_path + '.mp4', dpi=300, writer=FFMpegWriter(fps=5, codec='mpeg4'))

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + '.png')
    if show:
        plt.show()


def plot_2D_performance_with_prediction(fun: ConstrainedSyntheticTestFunction, X: Tensor, log_path: str, n_init: int, grid_size: int = 100, n_level: int = 10, 
    animation: bool = False, save_path: str = None, show: bool = True, plot_var: bool = True) -> None:
    """
    Plot performance with prediction over a 2D design space.
    """
    assert fun.dim == 2
    bounds = fun.bounds.numpy()
    locator = ticker.MaxNLocator(n_level)

    X0_grid, X1_grid = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    X_vec = np.hstack((X0_grid.reshape(-1, 1), X1_grid.reshape(-1, 1)))
    Y_vec, C_vec = fun(torch.from_numpy(X_vec))
    Y_vec, C_vec = Y_vec.numpy(), C_vec.numpy()
    valid_vec = (C_vec >= 0).all(axis=1)

    if plot_var:
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        n_row = 3
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        n_row = 2

    Y_grid = Y_vec.reshape(grid_size, grid_size)
    X_vec = unnormalize(X_vec, bounds)
    X0_grid, X1_grid = X_vec[:, 0].reshape(grid_size, grid_size), X_vec[:, 1].reshape(grid_size, grid_size)
    valid_grid = valid_vec.reshape(grid_size, grid_size).astype(float)
    Y_valid_grid = (Y_grid - Y_grid.min()) * valid_grid + Y_grid.min()

    # evaluated designs
    X = unnormalize(X.numpy(), bounds)
    n_batch = len(X) - n_init
    alphas = [0.1] * n_init
    for i, alpha in zip(range(n_batch), np.linspace(0.1, 1.0, n_batch + 1)[1:]):
        alphas.append(alpha)
    alphas = alphas[:len(X)]

    if fun._optimizers != None:
        optimizers = np.array(fun._optimizers)
        axes[0][2].scatter(optimizers[:, 0], optimizers[:, 1], marker='*', c='red', s=100)

    # ticks
    for i in range(n_row):
        for j in range(3):
            axes[i][j].set_xlabel('x1')
            axes[i][j].set_ylabel('x2')
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

    def onclick(event):
        x_click = np.array([event.xdata, event.ydata])
        x_click = normalize(x_click, bounds)[None, :]
        y_click, c_click = fun(torch.from_numpy(x_click))
        y_click, c_click = y_click.numpy()[0], c_click.numpy()[0]
        print(f'Objective: {y_click}, Constraint: {c_click >= 0.0}')

    fig.canvas.mpl_connect('button_press_event', onclick)

    n_curr = n_init
    Y_mean_grid_list, C_mean_grid_list, A_grid_list = [], [], []
    if plot_var:
        Y_var_grid_list, C_var_grid_list = [], []
    while True:
        log_path_curr = f'{log_path}_grid_{n_curr}.pkl'
        if not os.path.exists(log_path_curr):
            break
        with open(log_path_curr, 'rb') as fp:
            grid_data = pickle.load(fp)
        if 'Y_mean' in grid_data:
            Y_mean_grid_list.append(grid_data['Y_mean'])
        if 'C_mean' in grid_data:
            C_mean_grid_list.append(grid_data['C_mean'])
        if 'A' in grid_data:
            A_grid_list.append(grid_data['A'])
        if plot_var:
            if 'Y_var' in grid_data:
                Y_var_grid_list.append(grid_data['Y_var'])
            if 'C_var' in grid_data:
                C_var_grid_list.append(grid_data['C_var'])
        n_curr += 1

    cs = [[None for _ in range(3)] for _ in range(n_row)]
    cs[0][0] = axes[0][0].contourf(X0_grid, X1_grid, Y_grid, cmap='Blues', locator=locator)
    cs[0][1] = axes[0][1].contourf(X0_grid, X1_grid, valid_grid, cmap='Blues', locator=locator)
    axes[0][1].contour(X0_grid, X1_grid, valid_grid, [0.5])
    cs[0][2] = axes[0][2].contourf(X0_grid, X1_grid, Y_valid_grid, cmap='Blues', locator=locator)
    if len(Y_mean_grid_list) > 0:
        cs[1][0] = axes[1][0].contourf(X0_grid, X1_grid, Y_mean_grid_list[-1], cmap='Blues', locator=locator)
    if len(C_mean_grid_list) > 0:
        acc = np.isclose(np.round(C_mean_grid_list[-1]), valid_grid).sum() / (grid_size * grid_size)
        cs[1][1] = axes[1][1].contourf(X0_grid, X1_grid, C_mean_grid_list[-1], cmap='Blues', locator=locator)
        axes[1][1].contour(X0_grid, X1_grid, C_mean_grid_list[-1], [0.5])
    else:
        acc = None
    if len(A_grid_list) > 0:
        cs[1][2] = axes[1][2].contourf(X0_grid, X1_grid, A_grid_list[-1], cmap='Blues', locator=locator)
    if plot_var:
        if len(Y_var_grid_list) > 0:
            cs[2][0] = axes[2][0].contourf(X0_grid, X1_grid, Y_var_grid_list[-1], cmap='Blues', locator=locator)
        if len(C_var_grid_list) > 0:
            cs[2][1] = axes[2][1].contourf(X0_grid, X1_grid, C_var_grid_list[-1], cmap='Blues', locator=locator)
    colorbars = [[None for _ in range(3)] for _ in range(n_row)]
    for i in range(n_row):
        for j in range(3):
            if cs[i][j] is not None:
                colorbars[i][j] = fig.colorbar(cs[i][j], ax=axes[i][j])

    axes[0][0].set_title('Objective')
    axes[0][1].set_title('Constraint')
    axes[0][2].set_title('Constrained Objective')
    axes[1][0].set_title('Surrogate Objective (Mean)')
    if acc is not None:
        axes[1][1].set_title(f'Surrogate Constraint (Mean) (Accuracy: {acc})')
    else:
        axes[1][1].set_title('Surrogate Constraint (Mean)')
    axes[1][2].set_title('Surrogate Acquisition')
    if plot_var:
        axes[2][0].set_title('Surrogate Objective (Var)')
        axes[2][1].set_title('Surrogate Constraint (Var)')

    if not animation:
        for i in range(n_row):
            for j in range(3):
                axes[i][j].scatter(X[:, 0], X[:, 1], c='black', alpha=alphas[-1])
    else:
        def update(frame_number):
            len_X_frame = min(frame_number + n_init, len(X))
            for i in range(n_row):
                for j in range(3):
                    axes[i][j].cla()
                    axes[i][j].axis([X0_grid.min(), X0_grid.max(), X1_grid.min(), X1_grid.max()])
                    axes[i][j].set_xlabel('x1')
                    axes[i][j].set_ylabel('x2')
                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])
                    if colorbars[i][j] is not None:
                        colorbars[i][j].remove()

            cs[0][0] = axes[0][0].contourf(X0_grid, X1_grid, Y_grid, cmap='Blues', locator=locator)
            cs[0][1] = axes[0][1].contourf(X0_grid, X1_grid, valid_grid, cmap='Blues', locator=locator)
            axes[0][1].contour(X0_grid, X1_grid, valid_grid, [0.5])
            cs[0][2] = axes[0][2].contourf(X0_grid, X1_grid, Y_valid_grid, cmap='Blues', locator=locator)
            if len(Y_mean_grid_list) > 0:
                cs[1][0] = axes[1][0].contourf(X0_grid, X1_grid, Y_mean_grid_list[frame_number - 1], cmap='Blues', locator=locator)
            if len(C_mean_grid_list) > 0:
                acc = np.isclose(np.round(C_mean_grid_list[frame_number - 1]), valid_grid).sum() / (grid_size * grid_size)
                cs[1][1] = axes[1][1].contourf(X0_grid, X1_grid, C_mean_grid_list[frame_number - 1], cmap='Blues', locator=locator)
                axes[1][1].contour(X0_grid, X1_grid, C_mean_grid_list[frame_number - 1], [0.5])
            else:
                acc = None
            if len(A_grid_list) > 0:
                cs[1][2] = axes[1][2].contourf(X0_grid, X1_grid, A_grid_list[frame_number - 1], cmap='Blues', locator=locator)
            if plot_var:
                if len(Y_var_grid_list) > 0:
                    cs[2][0] = axes[2][0].contourf(X0_grid, X1_grid, Y_var_grid_list[frame_number - 1], cmap='Blues', locator=locator)
                if len(C_var_grid_list) > 0:
                    cs[2][1] = axes[2][1].contourf(X0_grid, X1_grid, C_var_grid_list[frame_number - 1], cmap='Blues', locator=locator)

            for i in range(n_row):
                for j in range(3):
                    if colorbars[i][j] is not None:
                        colorbars[i][j] = fig.colorbar(cs[i][j], ax=axes[i][j])
                        axes[i][j].scatter(X[:len_X_frame, 0], X[:len_X_frame, 1], c='black', alpha=alphas[-1])

            if fun._optimizers != None:
                optimizers = np.array(fun._optimizers)
                axes[0][2].scatter(optimizers[:, 0], optimizers[:, 1], marker='*', c='red', s=100)

            axes[0][0].set_title('Objective')
            axes[0][1].set_title('Constraint')
            axes[0][2].set_title('Constrained Objective')
            axes[1][0].set_title('Surrogate Objective (Mean)')
            if acc is not None:
                axes[1][1].set_title(f'Surrogate Constraint (Mean) (Accuracy: {acc})')
            else:
                axes[1][1].set_title('Surrogate Constraint (Mean)')
            axes[1][2].set_title(f'Surrogate Acquisition (Min: {"%.2E" % A_grid_list[frame_number - 1].min()}, Max: {"%.2E" % A_grid_list[frame_number - 1].max()})')
            if plot_var:
                axes[2][0].set_title('Surrogate Objective (Var)')
                axes[2][1].set_title('Surrogate Constraint (Var)')

        anim = FuncAnimation(fig, update, frames=n_batch + 1, interval=300)
        if save_path is not None:
            anim.save(save_path + '.mp4', dpi=300, writer=FFMpegWriter(fps=5, codec='mpeg4'))

    plt.tight_layout()
    if show:
        plt.show()
