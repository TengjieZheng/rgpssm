from typing import Tuple, Optional, List, Union
from types import SimpleNamespace

from torch import Tensor
from numpy import ndarray

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from utils import error_bar_plot

def subfig_plot(t, x_lst, label_lst, name_lst=None, ax_lst=None, color_lst=None,
                xlabel=None, flag_xticks=False, flag_label=False, legend_set={}, linewidth_lst=None, ls_lst=None, alpha_lst=None):
    """
    Args:
        t: (n,)
        x_lst: [(n, dx)]
    """

    n_col = x_lst[0].shape[-1]
    if ax_lst is None:
        fig, ax_lst = plt.subplots(1, n_col, figsize=(4 * n_col, 4))
        if n_col == 1:  # 保证 ax_lst 是可迭代的
            ax_lst = [ax_lst]
    for i in range(n_col):
        ax = ax_lst[i]
        for j, (x, l) in enumerate(zip(x_lst, label_lst)):
            c = color_lst[j] if color_lst is not None else 'black'
            lw = None if linewidth_lst is None else linewidth_lst[j]
            ls = None if ls_lst is None else ls_lst[j]
            aph = 0.7 if alpha_lst is None else alpha_lst[j]
            ax.plot(t, x[:, i], label=l, alpha=aph, color=c, linewidth=lw, ls=ls)
        if flag_label and i==0: ax.legend(**legend_set)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if name_lst is not None: ax.set_ylabel(name_lst[i])
        if not flag_xticks: ax.set_xticks([])
    plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.3, hspace=None)

    return ax_lst

def error_bar(ax_lst, t, x_lst, std_lst, color_lst=None, alpha=0.2):
    n_col = x_lst[0].shape[-1]
    for i in range(n_col):
        for j in range(len(x_lst)):
            c = color_lst[j] if color_lst is not None else 'yellow'
            error_bar_plot(t, x_lst[j][:, i], std_lst[j][:, i] * 1.96, label=None, alpha=alpha, ax=ax_lst[i], color=c)

    return ax_lst

def set_ylim(ax_lst, upper=0.1, lower=0.1, x_arr=None):
    """
    Args:
        ax_lst:
        upper:
        low:
        x_arr: (n, dx)
    """
    for i in range(len(ax_lst)):
        ax = ax_lst[i]
        if x_arr is None:
            Ylim = ax.set_ylim()
            ymin, ymax = Ylim
        else:
            x = x_arr[:, i]
            ymin, ymax = x.min(), x.max()
        d = ymax - ymin
        ax.set_ylim(ymin - d * lower, ymax + d * upper)

def area_plot(ax_lst, xc, color_l='g', color_r='r'):
    """plot background area"""
    for ax in ax_lst:
        Xlim = ax.set_xlim()
        ax.axvspan(Xlim[0], xc, color=color_l, alpha=0.06)
        ax.axvspan(Xlim[1], xc, color=color_r, alpha=0.07)
        ax.set_xlim(Xlim[0], Xlim[1])

def area_single(ax_lst, xl, xr, color='g'):
    """plot background area with a single color"""
    for ax in ax_lst:
        ax.axvspan(xl, xr, color=color, alpha=0.06)
        ax.set_xlim(xl, xr)