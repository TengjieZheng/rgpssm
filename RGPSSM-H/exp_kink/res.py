from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from trainer import Recorder
from utils import error_bar_plot, save_show, nMSE, Mnlp, get_color


def res_all():
    var_lst = [0.008, 0.08, 0.8]
    algo_lst = ['vcdt', 'envi', 'svmc', 'rgpssm_ekf', 'rgpssm_ukf', 'rgpssm_adf']
    name_lst = ['VCDT', 'EnVI', 'SVMC-GP', 'RGPSSM-H (EKF)', 'RGPSSM-H (UKF)', 'RGPSSM-H (ADF)']

    nmse_lst, nll_lst = [], []
    std_nmse_lst, std_nll_lst = [], []
    t_mean_lst, t_std_lst = [], []
    for algo in algo_lst:
        nmse_row, nll_row = [], []
        std_nmse_row, std_nll_row = [], []
        Data_lst, t_single_arr = model_plot(algo, flag_save=True, flag_show=False, filepath='./fig/')
        for i in range(3):
            d_lst = Data_lst[i]
            nmse_arr = np.array([d['nmse'] for d in d_lst])
            nll_arr = np.array([d['nll'] for d in d_lst])

            nmse_row.append(nmse_arr.mean())
            nll_row.append(nll_arr.mean())
            std_nmse_row.append(nmse_arr.std())
            std_nll_row.append(nll_arr.std())
        nmse_lst.append(nmse_row)
        nll_lst.append(nll_row)
        std_nmse_lst.append(std_nmse_row)
        std_nll_lst.append(std_nll_row)
        t_mean_lst.append(t_single_arr.mean())
        t_std_lst.append(t_single_arr.std())

    # Ouptut the latex table
    caption = ("Kink transition function learning performance using various methods across different levels of measurement noise "
               "($\sigma^2_m \in \{0.008,\, 0.08,\, 0.8\}$, from left to right), reported as mean $\pm$ standard deviation. "
               "The best results are highlighted in \\textbf{bold}, and the second-best results are \\underline{underlined}.")
    latex_code = generate_latex_table(nmse_lst, std_nmse_lst, nll_lst, std_nll_lst, name_lst, var_lst, t_mean_lst, t_std_lst, caption=caption)
    print(latex_code)


def model_plot(model, flag_save=False, flag_show=True, filepath='./'):

    fontsize = 8
    height = 3
    fig, axes = plt.subplots(1, 3, figsize=(height*3, height))

    D_lst = []
    t_lst = []
    for i, ax in enumerate(axes):
        d_lst, nmse_arr = get_data(model, i)
        data = d_lst[nmse_arr.argmin()]
        # data = d_lst[0]
        single_plot(ax, data, d_lst[0]['y'].ravel(), fontsize, isLegend=i==2)
        D_lst.append(d_lst)
        t_lst = t_lst + [d['t_train'] for d in d_lst]

    plt.tight_layout()

    save_show(flag_save, True, filepath + model + '_kink.pdf', fig)

    return D_lst, np.array(t_lst)

def get_data(model_name: str, i:int):
    var_list = [0.008, 0.08, 0.8]
    var = var_list[i]

    recorder = Recorder()

    nmse_lst = []
    d_lst = []
    for j in range(5):
        d, model = recorder.load(model_name, var, j)
        recorder.eval(d, model)

        nmse = nMSE(d['f_test'], d['f_mean'])
        nll = Mnlp(d['f_test'], d['f_mean'], d['f_std'])
        d['nmse'] = nmse
        d['nll'] = nll

        nmse_lst.append(nmse)
        d_lst.append(d)

    return d_lst, np.array(nmse_lst)

def single_plot(ax, data, y, fontsize=14, isLegend=True):
    # x, y = data['x'].ravel(), data['y'].ravel()
    x_test, f_test = data['x_test'], data['f_test']
    f_mean, f_std = data['f_mean'], data['f_std']
    in_ip, out_ip = data['inducing_in'], data['inducing_out']

    ax.plot(x_test, f_test, label='"kink" function', c='k')
    ax.scatter(y[:-1], y[1:], label='data', alpha=0.1, s=2, c='k')
    ax.plot(x_test, f_mean, label='GP mean', c=get_color('o'), alpha=1.)
    ax.plot(in_ip, out_ip, '*', label='inducing points', c=get_color('br'))
    error_bar_plot(x_test, f_mean, f_std * 1.96, label='95% Conf.', color=get_color('o'), alpha=0.2, ax=ax)

    ax.set_xlim([-3.15, 1.15])
    ax.set_ylim([-4.15, 2.1])
    ax.tick_params(axis='both', labelsize=fontsize)
    if isLegend:
        ax.legend(loc=0, fontsize=fontsize)

def generate_latex_table(
    nmse_lst, std_nmse_lst, nll_lst, std_nll_lst,
    algo_lst, var_lst, t_mean_lst, t_std_lst,
    refs=None, caption=None
):
    import numpy as np

    nmse_arr = np.array(nmse_lst)
    nll_arr = np.array(nll_lst)
    t_mean_arr = np.array(t_mean_lst)
    t_std_arr = np.array(t_std_lst)

    n_methods, n_vars = nmse_arr.shape

    # 找出每列最小值和第二小值的索引（NMSE, NLL）
    min_nmse_indices = np.argmin(nmse_arr, axis=0)
    second_min_nmse_indices = np.argsort(nmse_arr, axis=0)[1]

    min_nll_indices = np.argmin(nll_arr, axis=0)
    second_min_nll_indices = np.argsort(nll_arr, axis=0)[1]

    # 找出训练时间的最小值和次小值索引
    min_time_idx = np.argmin(t_mean_arr)
    second_min_time_idx = np.argsort(t_mean_arr)[1]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    {\footnotesize")  # 开始缩小字体范围
    lines.append("    " + r"\begin{tabular}{l" + "c" * len(var_lst) + "c}")
    lines.append(r"        \toprule")

    header = ["Method"] + [
        rf"$\sigma^2_m = {v}$ (nMSE $\downarrow$ $\mid$ MNLL $\downarrow$)" for v in var_lst
    ] + [r"Time (s) $\downarrow$"]
    lines.append("        " + " & ".join(header) + r" \\")
    lines.append(r"        \midrule")

    for i, (algo, nmse_row, std_nmse_row, nll_row, std_nll_row) in enumerate(
        zip(algo_lst, nmse_lst, std_nmse_lst, nll_lst, std_nll_lst)
    ):
        name = algo
        if refs is not None and refs[i]:
            name += rf"~\cite{{{refs[i]}}}"

        row_cells = []
        for j, (nmse, std_nmse, nll, std_nll) in enumerate(
            zip(nmse_row, std_nmse_row, nll_row, std_nll_row)
        ):
            is_min_mse = (i == min_nmse_indices[j])
            is_second_min_mse = (i == second_min_nmse_indices[j])

            is_min_nll = (i == min_nll_indices[j])
            is_second_min_nll = (i == second_min_nll_indices[j])

            # MSE
            mse_str = f"{nmse:.4f}$\\pm${std_nmse:.4f}"
            if is_min_mse:
                mse_str = r"\textbf{" + mse_str + r"}"
            elif is_second_min_mse:
                mse_str = r"\underline{" + mse_str + r"}"

            # NLL
            nll_str = f"{nll:.4f}$\\pm${std_nll:.4f}"
            if is_min_nll:
                nll_str = r"\textbf{" + nll_str + r"}"
            elif is_second_min_nll:
                nll_str = r"\underline{" + nll_str + r"}"

            content = mse_str + r" $\mid$ " + nll_str
            row_cells.append(content)

        # 训练时间列
        t_mean = t_mean_lst[i]
        t_std = t_std_lst[i]
        time_str = f"{t_mean:.2f}$\\pm${t_std:.2f}"
        if i == min_time_idx:
            time_str = r"\textbf{" + time_str + r"}"
        elif i == second_min_time_idx:
            time_str = r"\underline{" + time_str + r"}"
        row_cells.append(time_str)

        lines.append("        " + name + " & " + " & ".join(row_cells) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    }")  # 结束 \footnotesize 区块
    if caption:
        lines.append(r"    \caption{" + caption + "}")
    lines.append(r"    \label{tab:noise_comparison}")
    lines.append(r"\end{table}")

    return "\n".join(lines)



