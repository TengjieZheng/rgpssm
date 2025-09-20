from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import copy
import torch
import numpy as np
import time

from data.data_gen import load_data
from utils import save_pickle, load_pickle, Torch2Np, ToTensor, set_seed
from rgpssm_kink import RGPSSM_Kink
from envi.envi_kink import EnVI_Kink
from envi.vcdt_vgpssm_kink import VCDT_vGPSSM_Kink
from svmc.svmc_kink import SVMC_Kink

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')

def var_to_str(var):
    return f'{var:.3f}'

def f_kink(x):
    return 0.8 + (x + 0.2) * (1 - 5 / (1 + torch.exp(-2 * x)))


class Trainer():
    def __init__(self, model: str, var_noise: Union[float, List[float]], datapath:str):
        """
        Args:
            model: 'rgpssm_ekf', 'rgpssm_ukf', 'rgpssm_adf', 'envi', 'vcdt', 'vgpssm', 'svmc'
            var_noise:
        """
        if isinstance(var_noise, float):
            var_noise = [var_noise]

        self.model_name = model
        self.var_noise = var_noise
        self.data = load_data(datapath)

        self.recorder = Recorder()

    def run(self):
        set_seed(1)

        for var in self.var_noise:
            for i in range(5):
                data = self._data(var, i)

                self.model = self._init_model(self.model_name, var, data)

                start = time.time()
                self.model.train(data['y'])
                end = time.time()
                t_train = end - start
                data['t_train'] = t_train

                data_eval = self.recorder.eval(data, self.model)
                self.recorder.save(data_eval, self.model, self.model_name, var, i)

    def _init_model(self, name, var, data):

        if 'rgpssm' in name:
            model = RGPSSM_Kink(filter=name[-3:], var_noise=var)
        elif name == 'envi':
            model = EnVI_Kink(var_noise=var, device=device, ips=data['ips'])
        elif name == 'vcdt' or name =='vgpssm':
            model = VCDT_vGPSSM_Kink(name, var_noise=var, device=device, ips=data['ips'])
        elif name == 'svmc':
            model = SVMC_Kink(var_noise=var)
        else:
            raise ValueError('model must be one of: rgpssm_ekf, rgpssm_ukf, rgpssm_adf, envi, vcdt')

        return model

    def _data(self, var, i):

        vars = [0.8, 0.08, 0.008]
        keys_var = [f'var = {v:.3f}' for v in vars]
        vars = np.array(vars)
        d = np.abs(vars - var)
        idx = d.argmin()

        key = keys_var[idx] + f', i = {i}'
        d = self.data[key]

        return d



class Recorder():
    def __init__(self):
        self.log_path = './log/'

    def eval(self, data_xy, model):

        x_test = torch.linspace(-3.15, 1.15, 100)
        f_test = f_kink(x_test)

        f_mean, f_std = model.gp_pred(x_test)
        inducing_in, inducing_out = model.inducing_points()

        data = copy.deepcopy(data_xy)
        data['x_test'] = Torch2Np(x_test).ravel()
        data['f_test'] = Torch2Np(f_test).ravel()
        data['f_mean'] = Torch2Np(f_mean).ravel()
        data['f_std'] = Torch2Np(f_std).ravel()
        data['inducing_in'] = Torch2Np(inducing_in).ravel()
        data['inducing_out'] = Torch2Np(inducing_out).ravel()

        return data

    def save(self, data_eval, model, model_name, var, i):
        name = self.log_name(model_name, var, i)
        save_pickle([data_eval, model], name)

    def load(self, model, var, i):
        name = self.log_name(model, var, i)
        d = load_pickle(name)
        data_eval, model = d
        return data_eval, model

    def log_name(self, model, var, i):
        name0 = model + f'_var_' + var_to_str(var) + f'_{i}'
        name = self.log_path + name0 + '.pkl'

        return name