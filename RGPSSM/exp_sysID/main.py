import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import gpytorch
import copy

from method import *
from rgpssm import RGPSSM


class Data():
    """Get data"""
    def __init__(self, dataname='actuator'):
        self.dataname = dataname
        self.read_date(self.dataname)
        self.num_update = 0

    def read_date(self, dataname):
        DATA_DIR = 'Datasets'

        if dataname == 'actuator':
            raw_data = sio.loadmat(os.path.join(DATA_DIR, dataname))
            self.inputs = raw_data['u']
            self.outputs = raw_data['p']
        elif dataname == 'ballbeam':
            raw_data = np.loadtxt(os.path.join(DATA_DIR, 'ballbeam.dat'))
            self.inputs = raw_data[:, 0].reshape(-1, 1)
            self.outputs = raw_data[:, 1].reshape(-1, 1)
        elif dataname == 'drive':
            raw_data = sio.loadmat(os.path.join(DATA_DIR, 'drive.mat'))
            self.inputs = raw_data['u1']
            self.outputs = raw_data['z1']
        elif dataname == 'dryer':
            raw_data = np.loadtxt(os.path.join(DATA_DIR, 'dryer.dat'))
            self.inputs = raw_data[:, 0].reshape(-1, 1)
            self.outputs = raw_data[:, 1].reshape(-1, 1)
        elif dataname == 'gasfumace':
            raw_data = np.loadtxt(os.path.join(DATA_DIR, 'gas_furnace.csv'), skiprows=1, delimiter=',')
            self.inputs = raw_data[:, 0].reshape(-1, 1)
            self.outputs = raw_data[:, 1].reshape(-1, 1)

        self.inputs = self.normalize(self.inputs)
        self.outputs = self.normalize(self.outputs)
        self.dim_input = self.inputs.shape[1]
        self.dim_output = self.outputs.shape[1]

    def update(self):
        """Get a new data"""
        u, y = self.inputs[self.num_update, :], self.outputs[self.num_update, :]
        self.num_update += 1
        u, y = u.astype(np.float32), y.astype(np.float32)

        return u, y

    def normalize(self, x):

        y = np.copy(x).ravel()
        self.mean = np.mean(y)
        self.std = np.std(y)

        return (x - self.mean) / self.std


class Fun():
    def __init__(self, dim_latent):
        self.dim_latent = dim_latent

    def fun_meas(self, x):
        y = x.view(-1)[0].view(-1, 1)

        return y

    def fun_mean(self, z):
        f = torch.zeros((self.dim_latent, 1)) * z.view(-1)[0]

        return f

    def fun_tran(self, x, u, f):
        F = f.view(-1, 1)

        return F

    def fun_input(self, x, u):
        z = torch.cat((x.view(1, -1), u.view(1, -1)), dim=1)

        return z


def get_kernel(num_task, dim_GPin, var, ls):
    kernel = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dim_GPin),
                                              num_tasks=num_task, rank=0)  # Kxx \otimes V
    kernel.task_covar_module.var = torch.tensor(var).view(-1, 1)
    kernel.data_covar_module.lengthscale = torch.tensor(ls).view(-1, 1)

    return kernel


class single_sim():
    def __init__(self, param, seed=1):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Parameters
        self.data = Data(dataname=param['data'])
        rate_test = 0.5  # The proportion of the dataset to be used for the test set
        self.id_pre = int(self.data.inputs.shape[0]*rate_test)
        self.param = param

        # RGPSSM initialization
        dim_latent = param['dx']                # dimension of latent state
        x0 = np.zeros((dim_latent, 1))          # initial state mean
        P0 = np.eye(dim_latent) * 4             # initail state covariance
        Q = np.eye(dim_latent) * param['Q']     # process noise covariance
        R = np.eye(1) * param['R']              # measurement noise covariance

        kernel = get_kernel(num_task=dim_latent, dim_GPin=dim_latent + 1, var=[param['var']]*dim_latent, ls=param['ls'])
        self.gpssm = RGPSSM(x0, P0, Q, R, Fun(dim_latent), kernel, flag_chol=True,
                             jitter=param['jitter'], budget=param['budget'], eps_tol=param['eps'], num_opt_hp=param['num_opt_hp'], lr_hp=param['lr'])
        self.data_recorder = DataRecorder()

    def data_record_update(self):

        self.ls = self.gpssm.kernel.data_covar_module.lengthscale.detach().clone().numpy()
        self.var = self.gpssm.kernel.task_covar_module.var.detach().clone().numpy()

        data_name = ['u', 'y', 'x_est', 'std_est', 'x_pre', 'f_pre', 'std_f', 'std_x', 'ls', 'var']
        data_vec = [self.u, self.y, self.x_est, self.std_est, self.x_pre, self.f_pre, np.diag(self.var_f_pre) ** 0.5,
                    np.diag(self.var_x_pre) ** 0.5,
                    self.ls, self.var]
        self.data_recorder.data_add(data_name, data_vec)

    def run(self):

        try:
            # Learning
            for ii in range(self.data.inputs.shape[0]):
                # Get data
                self.u, self.y = self.data.update()
                # GPSSM update
                F, var_F, f, var_f = self.gpssm.predict(self.u)
                self.x_pre = Torch2Np(F)
                self.var_x_pre = Torch2Np(var_F)
                self.f_pre = Torch2Np(f)
                self.var_f_pre = Torch2Np(var_f)
                if ii < self.id_pre:
                    self.gpssm.correct(self.y)
                    self.x_est = Torch2Np(self.gpssm.x)
                    self.std_est = np.diag(Torch2Np(self.gpssm.P))**0.5
                else:
                    self.gpssm.eps_tol = 1e10
                if ii > self.data.inputs.shape[0]*0.1 and ii < self.id_pre:
                    self.gpssm.hyperparam_opt()
                # Record data
                self.data_record_update()
                if ii % 10 == 0:
                    print(f'ii = {ii}, ls = {self.ls}, var={self.var}, BV={self.gpssm.Z.shape[0]}')

            rmse = self.loss_eval()

        except RuntimeError:
            rmse = np.inf

        return rmse

    def loss_eval(self):
        d = self.data_recorder.database
        y = d['y'].ravel()[self.id_pre:]
        y_pre = d['x_pre'][:, 0].ravel()[self.id_pre:]
        std_pre = d['std_x'][:, 0].ravel()[self.id_pre:]

        rmse = self.metric(y, y_pre, std_pre)

        return rmse

    def metric(self, y, ypre, std_pre):

        e = y - ypre
        rmse = np.sqrt(np.mean(e ** 2)) * self.data.std

        return np.array(rmse).ravel()[0]

    def result_plot(self):
        # load data from data recorder
        dict = self.data_recorder.database
        keys_list = list(dict.keys())
        for name in keys_list:
            d = dict[name]
            if name in globals():
                print(f'{name} is already defined in the global namespace.')
            globals()[name] = d

        fig = plt.figure()
        plt.plot(y, label='$y$')
        plt.plot(x_pre[:, 0], label='$x_{1, \mathrm{pre}}$')
        error_bar_plot(None, x_pre[:, 0], std_x[:, 0] * 1.96, label='95% Conf.')
        plt.legend()

class All_data_learn():
    """Learning for all datasets"""

    def __init__(self):
        self.log_path = './log/log_all_data_learn.pkl'
        self.datalist = ['actuator', 'ballbeam', 'drive', 'dryer', 'gasfumace']

    def run(self):
        log_list = []
        for dataname in self.datalist:
            d_list = []
            for seed in [1,2,3,4,5]:
                obj = self.get_obj(dataname, seed)
                rmse = obj.run()
                d = obj.data_recorder.database
                d['rmse'] = rmse
                d['data_mean'] = obj.data.mean
                d['data_std'] = obj.data.std
                d['dataname'] = dataname
                d['id_pre'] = obj.id_pre
                d_list.append(d)
            log_list.append(d_list)

        self.save(log_list)

    def get_obj(self, dataname, seed=1):
        p = {}
        p['data'] = dataname
        p['jitter'] = 1e-8
        p['budget'] = 20
        p['eps'] = 1e-2
        p['dx'] = 4
        p['Q'] = 1e-4
        p['R'] = 1e-2
        p['lr'] = 5e-3
        p['ls'] = 4
        p['var'] = 8
        p['num_opt_hp'] = 3

        return single_sim(p, seed)

    def result_plot(self, flag_save=False, flag_show=True, filepath='./'):
        log_list = load_pickle(self.log_path)

        def single_plot(d_list):
            rmse_list = []
            for d in d_list:
                rmse_list.append(d['rmse'])
            rmse_list = np.array(rmse_list)
            id = np.argmin(rmse_list)

            log = d_list[id]
            data_mean, data_std = log['data_mean'], log['data_std']
            x_est, std_est = log['x_est'], log['std_est']
            y, x_pre, std_x, id_pre = log['y'], log['x_pre'], log['std_x'], log['id_pre']

            def recover(x):
                y = x * data_std + data_mean
                return y
            idx_seq = np.arange(x_est.shape[0])
            x_est_pre = np.vstack((x_est[:id_pre, :], x_pre[id_pre:, :]))
            std_est_pre = np.vstack((std_est[:id_pre, :], std_x[id_pre:, :]))

            fig = plt.figure(figsize=(8, 4))
            plt.plot(recover(y), label='measurement', color='k')
            plt.plot(idx_seq[:id_pre], recover(x_est[:id_pre, 0]), label='$x_{1, \mathrm{filtered}}$', color='#FF8C00', ls=(0,(5, 3)))
            plt.plot(idx_seq[id_pre:], recover(x_pre[id_pre:, 0]), label='$x_{1, \mathrm{predicted}}$', color='#FF8C00')
            error_bar_plot(None, recover(x_est_pre[:, 0]), data_std * std_est_pre[:, 0] * 1.96, label='95% Conf.', color='#FFA500', alpha=0.15)
            plt.legend(prop = { "size": 8 })

            ymax = np.max(recover(y))
            ymin = np.min(recover(y))
            Dy = ymax - ymin
            ylim1, ylim2 = ymin - 0.1*Dy, ymax + 0.1*Dy
            plt.ylim(ylim1, ylim2 )

            rmse = rmse_list.mean()
            std = rmse_list.std()
            plt.title(f'RMSE = {rmse:.3f} $\pm$ {std:.3f}', fontsize=10)
            print(f'Predition RMSE - ' + log['dataname'] + f' : {rmse:.3f} ± {std:.3f}')

            save_show(flag_save, flag_show, filepath+'sysID_'+log['dataname']+'.pdf', fig)

        for log in log_list:
            single_plot(log)


    def save(self, log_list):
        save_pickle(log_list, self.log_path)

    def load(self):
        return load_pickle(self.log_path)


if __name__ == '__main__':
    test = All_data_learn()
    test.run()
    test.result_plot(flag_save=True, filepath='./res/')
    plt.show()

