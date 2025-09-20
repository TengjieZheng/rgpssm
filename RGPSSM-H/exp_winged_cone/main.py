from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from trainer import Trainer
from res import res_all



if __name__ == '__main__':
    alg_lst = ['rgpssm', 'rgpssm_h_adf']

    for alg in alg_lst:
        train = Trainer(name=alg)
        train.run(tend=60, t_ctrl=40)

    alg_lst = ['rgpssm', 'rgpssm_h_adf']
    res_all(alg_lst)

    plt.show()