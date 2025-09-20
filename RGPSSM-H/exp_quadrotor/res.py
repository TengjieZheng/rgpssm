from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from trainer import Recorder
from utils import save_show, error_bar_plot, get_color, get_color_sys
from utils_res import subfig_plot, error_bar, set_ylim, area_plot


def res_all(alg_lst=None):
    name_map = {
        'rgpssm': 'RGPSSM',
        'rgpssm_h_ekf': 'RGPSSM-H',
        'rgpssm_h_ukf': 'RGPSSM-H',
        'rgpssm_h_adf': 'RGPSSM-H',
        'ekf': 'EKF',
    }
    if alg_lst is None:
        alg_lst = name_map.keys()


    color_gt = 'gray'
    color_map = {
        'rgpssm': get_color('b'),
        'rgpssm_h_ekf': get_color('r'),
        'rgpssm_h_ukf': get_color('g'),
        'rgpssm_h_adf': get_color('o'),
        'ekf': get_color('b'),
        'train': get_color_sys(1),
        'pred': get_color_sys(2),
    }

    state_plot(alg_lst, color_gt, name_map, color_map)
    hyperparam_plot(alg_lst, color_gt, name_map, color_map)

    # prediction error
    time_map = {}
    metric = {}
    for alg in alg_lst:
        _, data = Recorder.load(name=alg)
        mse, nll = Recorder.eval(data)
        m = {'mse': mse, 'nll': nll}
        metric[alg] = m
        time_map[alg] = data['t_sim']
    text = metric_to_latex(metric, name_map, time_map, label='tab:learn_quadrotor',
                           caption='Prediction performance and total learning time for quadrotor dynamics using EKF and RGPSSM-H. The column "Time" denotes the total runtime for processing 30 seconds of measurements.')
    print(text)

def state_plot(alg_lst, color_gt, name_map, color_map):
    fig, axs = plt.subplots(3, 3, figsize=(17, 6))

    # Ground truth
    _, data = Recorder.load(name=alg_lst[0])
    subfig_plot(data['t'], [data['p']], label_lst=['measurement'],
                name_lst=['Position $x$ [m]', 'Position $y$ [m]', 'Position $z$ [m]'], ax_lst=axs[0], color_lst=[color_gt], linewidth_lst=[2])
    subfig_plot(data['t'], [data['v']], label_lst=['reference'], flag_label=True, legend_set={'ncol':2, 'loc': 'upper left', 'fontsize':8},
                name_lst=['Velocity $v_x$ [m/s]', 'Velocity $v_y$ [m/s]', 'Velocity $v_z$ [m/s]'], ax_lst=axs[1], color_lst=[color_gt], linewidth_lst=[2])
    subfig_plot(data['t'], [data['a']], label_lst=['reference'], flag_label=True, legend_set={'ncol':2, 'loc': 'upper left', 'fontsize':8},
                name_lst=['Acceleration $a_x$ [m/$\mathrm{s}^2$]', 'Acceleration $a_y$ [m/$\mathrm{s}^2$]', 'Acceleration $a_z$ [m/$\mathrm{s}^2$]'], ax_lst=axs[2], color_lst=[color_gt], xlabel='Time [s]',
                flag_xticks=True, linewidth_lst=[2])

    for alg in alg_lst:
        _, data = Recorder.load(name=alg)
        # Position
        subfig_plot(data['t'], [data['x_est'][:, :3]],
                    label_lst=[name_map[alg]], ax_lst=axs[0], color_lst=[color_map[alg]], flag_label=True,
                    legend_set={'ncol':2, 'loc': 'upper left', 'fontsize':8}, alpha_lst=[0.5])
        # error_bar(axs[0], data['t'], [data['x_est'][:, :3]],
        #           [data['std_est'][:, :3]], color_lst=[color_map[alg]], alpha=0.12)
        # Velocity
        subfig_plot(data['t'], [data['x_est'][:, 3:6]],
                    label_lst=[name_map[alg]], ax_lst=axs[1], color_lst=[color_map[alg]], alpha_lst=[0.7])
        # error_bar(axs[1], data['t'], [data['x_est'][:, 3:6]],
        #           [data['std_est'][:, 3:6]], color_lst=[color_map[alg]], alpha=0.12)
        # Acceleration
        subfig_plot(data['t'], [data['f_pre']],
                    label_lst=[name_map[alg]], ax_lst=axs[2], color_lst=[color_map[alg]], flag_xticks=True, alpha_lst=[0.7])
        # error_bar(axs[2], data['t'], [data['f_pre']],
        #           [data['std_f']], color_lst=[color_map[alg]], alpha=0.12)


    axs[0][0].text(8., -0.93, 'learn', size=10, color=color_map['train'])
    axs[0][0].text(35., -0.93, 'predict', size=10, color=color_map['pred'])
    set_ylim(ax_lst=axs[0], upper=0.4, lower=0.25, x_arr=data['p'])
    set_ylim(ax_lst=axs[1], upper=0.2, lower=0.2, x_arr=data['v'])
    set_ylim(ax_lst=axs[2], upper=0.2, lower=0.2, x_arr=data['a'])
    for i in range(3):
        area_plot(axs[i], data['t_pre'], color_map['train'], color_map['pred'])
    plt.tight_layout()
    save_show(flag_save=True, flag_show=True, filename='./fig/learn_quadrotor.pdf', fig=fig)

    # 3D trajectory
    p = data['p']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(p[0, 0], p[0, 1], p[0, 2], label='starting position', s=20, alpha=0.9, color='green')
    ax.scatter3D(p[-1, 0], p[-1, 1], p[-1, 2], label='end position', s=20, alpha=0.9, color='#F46F44')
    ax.plot3D(p[:, 0], p[:, 1], p[:, 2], label='trajectory', alpha=0.5, linewidth=1.5, color='b')
    ax.set_xlabel('Position $x$ [m]')
    ax.set_ylabel('Position $y$ [m]')
    ax.set_zlabel('Position $z$ [m]')
    ax.legend(loc='upper right')
    save_show(fig=fig, flag_save=True, flag_show=True, filename='./fig/traj.pdf')


def hyperparam_plot(alg_lst, color_gt, name_map, color_map):
    _, data = Recorder.load(name=alg_lst[0])
    dz = 8
    fig, axs = plt.subplots(nrows=3, ncols=dz, figsize=(15, 7))
    for i in range(3):
        for alg in alg_lst:
            if 'rgpssm' in alg:
                _, data = Recorder.load(name=alg)
                if alg == 'rgpssm':
                    ls = data['ls']
                else:
                    ls = data['ls'][:, dz*i:dz*(i+1)]
                subfig_plot(data['t'], [ls], label_lst=[name_map[alg]], ax_lst=axs[i], color_lst=[color_map[alg]], flag_label=i==0)

    plt.tight_layout()

def metric_to_latex(metric, name_map, time_map, channel_names=None, caption=None, label=None):
    """
    Convert metric dict into LaTeX table code with boldface for the best values.
    Args:
        metric: dict, structure like {alg: {'mse': (3,), 'nll': (3,)}}
        name_map: dict, {alg: str}, mapping algorithm keys to display names (LaTeX-safe)
        time_map: dict, {alg: float}, runtime (seconds) for each algorithm
        channel_names: list[str], channels names, e.g. ["$a_x$", "$a_y$", "$a_z$"]
        caption: table caption (string)
        label: table label (string)
    """
    algs = list(metric.keys())
    n_channel = len(metric[algs[0]]['mse'])

    if channel_names is None:
        channel_names = ['$a_x$', '$a_y$', '$a_z$']

    # Compute best (minimum) values across algorithms
    best_mse = [min(metric[alg]['mse'][i] for alg in algs) for i in range(n_channel)]
    best_nll = [min(metric[alg]['nll'][i] for alg in algs) for i in range(n_channel)]

    # Build column headers
    header_parts = [f"{name} (nMSE $\\mid$ MNLL)$\\downarrow$" for name in channel_names]
    header = "Method & " + " & ".join(header_parts) + " & Time (s)$\\downarrow$ \\\\"

    # Build rows
    rows = []
    for alg in algs:
        mse = metric[alg]['mse']
        nll = metric[alg]['nll']
        parts = []
        for i in range(n_channel):
            mse_str = f"{mse[i]:.4f}"
            nll_str = f"{nll[i]:.4f}"
            # Bold if best
            if mse[i] == best_mse[i]:
                mse_str = f"\\textbf{{{mse_str}}}"
            if nll[i] == best_nll[i]:
                nll_str = f"\\textbf{{{nll_str}}}"
            parts.append(f"{mse_str} $\\mid$ {nll_str}")
        time_val = time_map.get(alg, float('nan'))
        row = f"{name_map[alg]} & " + " & ".join(parts) + f" & {time_val:.2f} \\\\"
        rows.append(row)

    rows_str = "\n        ".join(rows)

    # Build LaTeX table
    latex = (
        "\\begin{table}[htbp]\n"
        "    \\centering\n"
        "    \\footnotesize\n"
        "    \\begin{tabular}{l" + "c"*(n_channel+1) + "}\n"
        "        \\toprule\n"
        f"        {header}\n"
        "        \\midrule\n"
        f"        {rows_str}\n"
        "        \\bottomrule\n"
        "    \\end{tabular}\n"
        "    \\normalsize\n"
    )
    if caption:
        latex += f"    \\caption{{{caption}}}\n"
    if label:
        latex += f"    \\label{{{label}}}\n"
    latex += "\\end{table}"

    return latex


