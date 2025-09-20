from typing import Tuple, Optional, List, Union
from torch import Tensor
from numpy import ndarray

from functools import wraps
import sys
import time
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pickle
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataRecorder():
    def __init__(self, capality=None, flag_save=False, filepath = r'./'):
        self.flag_save = flag_save  # 若为真，则会在database第一次满了和以后每更新一轮的时候进行保存
        self.filepath = filepath    # 数据保存的路径

        self.database = {}          # 数据库
        self.empty_flag = True
        self.full_flag = False
        self.number_data = 0        # 接收到的数据总数，不等于数据库中的数据数量
        self.number_update = 0      # database更新完一轮的次数
        self.capality = capality    # 数据库的容量

    def data_add(self, data_name, data_vector):
        if self.empty_flag == True:
            for name, data in zip(data_name, data_vector):
                self.database[name] = np.array(data).reshape((1, -1))
            self.empty_flag = False
        else:
            if self.full_flag == False:
                for name, data in zip(data_name, data_vector):
                    self.database[name] = np.vstack(( self.database[name], np.array(data).reshape((1, -1)) ))
            else:
                for name, data in zip(data_name, data_vector):
                    self.database[name] = np.vstack(( self.database[name], np.array(data).reshape((1, -1)) ))
                    self.database[name] = self.database[name][1:, :]  # 先进先出

        self.number_data += 1

        flag_updata_finish = False  # database是否更新完一轮
        if not self.capality == None:
            if self.number_data >= self.capality:
                self.full_flag = True

            if np.abs(self.number_data % self.capality) < 1e-2 and self.number_data > 0:
                flag_updata_finish = True
                self.number_update += 1

            if flag_updata_finish and self.flag_save:
                # 保存当前的database
                filename = self.filepath + str(self.number_update) + '.txt'
                self.dictionary_save(self.database, filename)

        return flag_updata_finish

    def dictionary_save(self, dict, filename):
        # 字典变量保存
        keys_list = list(dict.keys())
        file = open(filename, 'w')  # 写入会覆盖

        for key in keys_list:
            s = key + ' '
            file.write(s)

            data = dict[key].ravel().tolist()
            s = str(data).replace('[', '').replace(']', '')  # 去除[],这两行代码按数据不同，可以选择
            s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
            file.write(s)

        file.close()
        print("dictionary save successfully")

    def dictionary_read(self, filename):
        # 字典变量读取
        file = open(filename, 'r')

        dict = {}
        arr = []
        while 1:
            s = file.readline()
            if s == '':
                break
            s = s.splitlines()[0]  # 去掉末尾的换行符
            s = s.split(' ')  # 按空格分割
            index = 0
            for ss in s:
                if index == 0:
                    key = ss
                else:
                    if ss != '':
                        arr.append(float(ss))
                index += 1
            dict[key] = np.array(arr)
            arr = []

        file.close()

        return dict

def timing(func):
    @wraps(func)  # 保留func的属性
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        stop = time.time()
        print('%r took %f s\n' % (func.__name__, stop - start))
        sys.stdout.flush()  # 立即刷新输出
        return res

    return wrapper

# Data type transformation methods
def Np2Num(x):
    return np.array(x).ravel()[0]

def ToTensor(x: Union[ndarray, List, Tensor], device=None, dtype=torch.float32, view:Optional[Union[List, Tuple]]=None):
    """to tensor"""
    if x is None:
        y = x
    else:
        if isinstance(x, torch.Tensor):
            y = x
        else:
            y = torch.tensor(x)

        if dtype is not None:
            y = y.to(dtype)

        if device is not None:
            return y.to(device)

    if view is not None and y is not None:
        y = y.view(*view)
    return y

def Torch2Np(x):
    return x.detach().to('cpu').numpy()

# File methods
def save_pickle(data, file_name, flag_show=True):
    # 将data保存成pkl文件
    f = open(file_name, 'wb')
    pickle.dump(data, f)
    f.close()
    if flag_show:
        print(file_name + ' save successfully')

def load_pickle(file_name, flag_show=True):
    # pkl文件读取
    f = open(file_name, 'rb+')
    data = pickle.load(f)
    f.close()
    if flag_show:
        print(file_name + ' load successfully')

    return data

# Plot methods
def save_show(flag_save, flag_show, filename, fig, dpi=None):
    if flag_save:
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
    if not flag_show:
        plt.close(fig)

def set_legend(ax, rate_margin=0.15, ncol=5, fontsize=8, loc='upper right'):
    """Set the format of legend
    Arguments:
        rate_margin : rate of the top margin [float]
        ncol : number of the columns of the legend [int]
        fontsize : font size of the legend [int]
        loc : location of the legend [str]
    """
    Ylim = ax.set_ylim()
    d = Ylim[1] - Ylim[0]
    ax.set_ylim(Ylim[0], Ylim[1] + rate_margin * d)
    ax.legend(loc=loc, fontsize=fontsize, ncol=ncol)

def error_bar_plot(t, x, e, label='e', alpha=0.2, color='yellow', ax=None):
    # 误差带绘制
    x = x.ravel()
    e = e.ravel()
    if t is None:
        t = np.arange(x.size).ravel()
    else:
        t = t.ravel()

    if ax is None:
        plt.fill_between(t, x - e, x + e, alpha=alpha, color=color, label=label)
    else:
        ax.fill_between(t, x - e, x + e, alpha=alpha, color=color, label=label)

def meshgrid_cal(input, f, num_output):
    # 功能：进行meshgrad矩阵的函数计算
    # 输入：
    # input 一个列表，每个元素是输入一个维度的数据
    # f 函数句柄
    # num_output f输出的个数
    # 输出：
    # output 若num_output为1则是一个矩阵，若为=大于1则是一个列表，列表包含num_output个矩阵

    shape_data = input[0].shape
    if num_output == 1:
        output = np.zeros_like(input[0]).ravel()
    elif num_output > 1:
        output0 = np.zeros_like(input[0]).ravel()
        output = []
        for ii in range(num_output):
            output.append(np.copy(output0))

    for ii in range(input[0].size):
        input_array = np.zeros((len(input), 1)).ravel()
        for jj in range(len(input)):
            input_array[jj] = input[jj].ravel()[ii]
        if num_output == 1:
            output[ii] = f(input_array)
        elif num_output > 1:
            Out = f(input_array)
            for kk in range(num_output):
                output[kk][ii] = Out[kk]

    if num_output == 1:
        output = output.reshape(shape_data)
    elif num_output > 1:
        for kk in range(num_output):
            output[kk] = output[kk].reshape(shape_data)

    return output

def surface_plot(ax, X, Y, Z, alpha=1, label='', cmap=None, color=None):
    # 曲面图绘制
    surf = ax.plot_surface(X, Y, Z, alpha=alpha, label=label, cmap=cmap, color=color)
    # 解决图例报错
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    # plt.legend()

    return surf

def nMSE(x: ndarray, x_pre: ndarray):
    """x (n,)"""
    x_, x_pre_ = x.ravel(), x_pre.ravel()
    var_x = np.var(x_)
    mse = np.mean((x_-x_pre_)**2)
    nmse = mse / var_x
    return nmse

def Mnlp(x: ndarray, x_pre: ndarray, std: ndarray):
    """Mean Negative Log Probability
    Args:
        x: (n,)
        x_pre: (n,)
        std: (n)
    Returns:
        l: (1,)
    """
    x, x_pre, std = x.ravel(), x_pre.ravel(), std.ravel()

    log_std = 2 * np.log(std)
    log_pi = np.log(2 * np.pi)
    e_ = (x - x_pre) / std

    mnlp = np.mean(e_**2 + log_std + log_pi) * 0.5

    return mnlp



if __name__ == '__main__':
    pass