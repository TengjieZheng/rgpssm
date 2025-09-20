from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from data.data_gen import generate_data
from res import model_plot, res_all
from trainer import Trainer

if __name__ == '__main__':
    # Generate data
    datapath = './data/dataset/data_kink.pkl'
    # generate_data(datapath)

    # Model training
    model = 'rgpssm_adf'            # select a model: ’vcdt‘, 'envi', 'svmc', 'rgpssm_ekf', 'rgpssm_ukf', 'rgpssm_adf'
    var_lst = [0.008, 0.08, 0.8]    # noise variance

    trainer = Trainer(model=model, var_noise=var_lst, datapath=datapath)
    trainer.run() # train model

    # Result
    model_plot(model=model, filepath='./fig/')  # res of a single model
    res_all()                                   # res of all models
    plt.show()
