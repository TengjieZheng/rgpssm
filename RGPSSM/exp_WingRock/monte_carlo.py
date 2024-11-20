import gpytorch
import matplotlib.pyplot as plt
import numpy as np

from method import *
from rgpssm import RGPSSM
from wing_rock import WingRock
from get_hyperparameter import RefHyperparam


class ObjectWingRock():
    """Load wing rock data"""

    def __init__(self):
        self.log = load_pickle('./log/wr.pkl')
        self.wr = self.log.obj                          # wing rock module
        self.sigma_noise = self.log.sigma_noise         # standard derivation of measurement noise
        self.data = self.log.data_recorder.database     # state data
        self.dt = self.wr.dt                            # time step

        self.num_data = self.data['t'].size
        self.num_update = 0

    def update(self):
        if self.num_update < self.num_data:
            self.delta, self.y = self.data['delta'][self.num_update, 0], self.data['y'][self.num_update, 0]     # control input and measurement
            self.t = self.data['t'][self.num_update, 0]                                                         # time
            self.theta = self.data['theta'][self.num_update, 0]                                                 # roll angle
            self.theta_d = self.data['theta_d'][self.num_update, 0]                                             # roll angle command
            self.p = self.data['p'][self.num_update, 0]                                                         # roll rate
            self.dot_p = self.data['dot_p'][self.num_update, 0]                                                 # roll acceleration
            self.Delta = self.data['Delta'][self.num_update, 0]                                                 # model uncertainty

        self.num_update += 1

        return self.delta, self.y

class Fun():

    def __init__(self, wr):
        self.wr = wr

    def fun_meas(self, x):
        y = x.view(-1)[0].view(-1, 1)

        return y

    def fun_mean(self, z):
        f = 0.0 * z.view(-1)[0].view(-1, 1)

        return f

    def fun_tran(self, x, u, f):

        x = x.view(-1, 1)
        dot_p = f.view(-1, 1) + self.wr.L * u.view(-1, 1)
        dot_x = torch.cat((x[1, :].view(-1, 1), dot_p), dim=0)
        F = x + dot_x * self.wr.dt

        return F

    def fun_input(self, x, u):
        z = x.view(1, -1)

        return z

class Learn():
    def __init__(self, param):
        self.param = param
        np.random.seed(1)
        torch.manual_seed(1)
        self.obj = ObjectWingRock()

        x0 = torch.zeros((2, 1))                    # mean of initial state
        P0 = torch.eye(2)                           # covariance of initial state
        Q = torch.eye(2) * 1e-30                    # process noise covariance
        Q[1, 1] = 1e-2 * self.obj.dt
        R = torch.eye(1) * self.obj.sigma_noise**2  # measurement noise covariance

        num_task, dim_GP_input = 1, 2               # dimension of GP output, dimension of GP input
        var_task=[self.param['var']]
        lengthscale=[self.param['ls']]*2
        kernel = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dim_GP_input),
                                                       num_tasks=num_task, rank=0) # Kxx \otimes V
        kernel.task_covar_module.var = torch.tensor(var_task).view(-1, 1)
        kernel.data_covar_module.lengthscale = torch.tensor(lengthscale).view(-1, 1)

        self.gpssm = RGPSSM(x0, P0, Q, R, Fun(self.obj.wr), kernel, flag_chol=True,
                              jitter=1e-4, budget=20, eps_tol=1e-3, num_opt_hp=param['num_opt_hp'], lr_hp=param['lr'], Qu=param['Qu'])

        self.data_recorder = DataRecorder()

        self.t_learn = 0.
    def data_record_update(self):
        self.ls = self.gpssm.kernel.data_covar_module.lengthscale.detach().clone().numpy()
        self.var = self.gpssm.kernel.task_covar_module.var.detach().clone().numpy()
        self.data_name = ['t', 'theta', 'theta_d', 'p', 'dot_p','Delta', 'delta', 'y', 'x_est', 'x_pre', 'f_pre', 'var_f', 'var_x',
                          'ls', 'var']
        self.data_vector = [self.obj.t, self.obj.theta, self.obj.theta_d, self.obj.p, self.obj.dot_p, self.obj.Delta, self.delta, self.y,
                            self.x_est, self.x_pre, self.f_pre, np.diag(self.var_f_pre), np.diag(self.var_x_pre),
                            self.ls, self.var]
        self.data_recorder.data_add(self.data_name, self.data_vector)

    def run(self, flag_index=False):
        tend = self.param['tend']
        tpre = self.param['tend']
        for ii in range(int(tend/self.obj.dt)):
            self.delta, self.y = self.obj.update()

            # Prediction step
            start = time.time()
            F, var_F, f, var_f = self.gpssm.predict(Np2Torch(self.delta).view(-1))
            self.x_pre = Torch2Np(F)
            self.var_x_pre = Torch2Np(var_F)
            self.f_pre = Torch2Np(f)
            self.var_f_pre = Torch2Np(var_f)

            if self.obj.t >= tpre:
                self.gpssm.eps_tol = 1e10

            # Correction step
            if self.obj.t < tpre:
                self.gpssm.correct(Np2Torch(self.y))
            self.x_est = Torch2Np(self.gpssm.x)

            end = time.time()
            self.t_learn += end - start

            # Hyperparameter optimization
            if ii > 100 and self.obj.t < tpre:
                self.gpssm.hyperparam_opt()

            self.data_record_update()
            if ii%100 == 0 and flag_index:
                num_BV = self.gpssm.Z.shape[0]
                print(f'ii = {ii}, BV = {num_BV}')
                print(f'l = {self.ls}, var = {self.var}')

        return self.eval()

    def eval(self):
        'Prediction RSME'
        dict = self.data_recorder.database
        idx = int(dict['t'].size*0.75)
        Delta = dict['Delta'][idx:, 0]
        fpre = dict['f_pre'][idx:, 0]

        return self.metric(Delta, fpre)

    def metric(self, y, ypre, std_pre=None):

        e = y - ypre
        l = np.sqrt(np.mean(e ** 2))

        return np.array(l).ravel()[0]

    def result_plot(self):
        # load data from data recorder
        dict = self.data_recorder.database
        keys_list = list(dict.keys())
        for name in keys_list:
            d = dict[name]
            globals()[name] = d

        color = '#350E62'

        # profiles of states
        fig = plt.figure(figsize=(17, 4))
        ax = fig.add_subplot(141)
        plt.plot(t, theta_d, label=r'$\theta_d$')
        plt.plot(t, theta, label=r'$\theta$')
        plt.plot(t, y, label=r'$y$')
        plt.plot(t, x_pre[:, 0], label=r'$\hat\theta$')
        error_bar_plot(t, x_pre[:, 0], np.sqrt(var_x[:, 0]) * 1.96, color='y', alpha=0.2, label='$e_x$')
        plt.legend()
        plt.xlabel('Time [s]')
        ax = fig.add_subplot(142)
        plt.plot(t, p, label=r'$p$')
        plt.plot(t, x_pre[:, 1], label=r'$\hat p$')
        # plt.plot(t, f_pre[:, 0], label=r'$\hat p_{gp}$')
        error_bar_plot(t, x_pre[:, 1], np.sqrt(var_x[:, 1]) * 1.96, color='y', alpha=0.2, label='$e_x$')
        # error_bar_plot(t, f_pre[:, 0], np.sqrt(var_f[:, 0]) * 1.96, color='#350E62', alpha=0.2, label='$e_f$')
        plt.legend()
        plt.xlabel('Time [s]')
        ax = fig.add_subplot(143)
        # plt.plot(t, dot_p, label='$\dot p$')
        plt.plot(t, Delta, label='$\Delta$')
        plt.plot(t, f_pre[:, 0], label=r'$\hat{\dot p}_{gp}$')
        error_bar_plot(t, f_pre[:, 0], np.sqrt(var_f[:, 0]) * 1.96, color='#350E62', alpha=0.2, label='$e_f$')
        plt.legend()
        plt.xlabel('Time [s]')
        ax = fig.add_subplot(144)
        plt.plot(t, delta, label='$\delta$')
        plt.legend()
        plt.xlabel('Time [s]')
        plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.3, hspace=None)

        fig = plt.figure(figsize=(6, 7))
        ax = fig.add_subplot(311)
        plt.plot(t, ls[:, 0], label='$l_{x_1}$')
        plt.legend()
        ax = fig.add_subplot(312)
        plt.plot(t, ls[:, 1], label='$l_{x_2}$')
        plt.legend()
        ax = fig.add_subplot(313)
        plt.plot(t, var[:, 0], label='$\sigma^2$')
        plt.legend()
        plt.xlabel('Time [s]')
        plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.3, hspace=None)
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry("+100+100")  # 调整位置到 (100, 100)

        # learning effects
        theta_arr = np.linspace(-3.5, 3.5, 30)
        p_arr = np.linspace(-3.5, 3.5, 30)
        theta_mg, p_mg = np.meshgrid(theta_arr, p_arr)

        def Delta_true(x):
            return self.obj.wr.uncertainty(x.ravel()[0], x.ravel()[1])

        f_true_mg = meshgrid_cal([theta_mg, p_mg], Delta_true, num_output=1)

        def Delta_pre(x):
            f, var = self.gpssm.GP_predict(x.reshape(1, -1))
            f, var = Torch2Np(f), Torch2Np(var)

            return Np2Num(f), Np2Num(var)

        f_pre_mg, Var_mg = meshgrid_cal([theta_mg, p_mg], Delta_pre, num_output=2)

        cmap = 'plasma'
        alpha_3D = 0.4
        fig = plt.figure(figsize=(15, 4))
        ax = fig.add_subplot(131, projection='3d')
        surface_plot(ax, theta_mg, p_mg, f_true_mg, alpha=alpha_3D, label='$\dot p_{true}$')  # gpr预测图
        surface_plot(ax, theta_mg, p_mg, f_pre_mg, alpha=alpha_3D, label='$\dot p_{pre}$')  # gpr预测图
        plt.legend()
        ax.set_xlabel(r'$\theta$ [deg]')
        ax.set_ylabel(r'$p$ [deg/s]')
        plt.title('Predictive effects')

        ax = fig.add_subplot(132)
        e_mg = np.abs(f_true_mg - f_pre_mg)
        surf = plt.contourf(theta_mg, p_mg, e_mg, 20, cmap=plt.get_cmap(cmap))
        BV = Torch2Np(self.gpssm.Z)
        plt.scatter(theta, p, label='sample', marker='o', facecolor='none', edgecolors='w', s=1.5, alpha=0.8, linewidths=0.10)
        plt.scatter(BV[:, 0], BV[:, 1], label=r'inducing point', marker='D', s=30, alpha=0.9, facecolors='none', edgecolors='g', linewidths=1.8)
        plt.legend()
        ax.set_xlabel(r'$\theta$ [deg]')
        ax.set_ylabel(r'$p$ [deg/s]')
        cbar = fig.colorbar(surf, shrink=1, aspect=20)
        plt.title('Predictions errors')

        ax = fig.add_subplot(133)
        Interval = 1.96 * Var_mg ** 0.5
        surf = plt.contourf(theta_mg, p_mg, Interval, 20, cmap=plt.get_cmap(cmap))
        plt.scatter(theta, p, label='sample', marker='o', facecolor='none', edgecolors='w', s=1.5, alpha=0.8,  linewidths=0.10)
        plt.scatter(BV[:, 0], BV[:, 1], label=r'inducing point', marker='D', s=30, alpha=0.9, facecolors='none', edgecolors='g', linewidths=1.8)
        plt.legend()
        ax.set_xlabel(r'$\theta$ [deg]')
        ax.set_ylabel(r'$p$ [deg/s]')
        cbar = fig.colorbar(surf, shrink=1, aspect=20)
        plt.title('95% confidence interval')
        plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.3, hspace=None)

        self.local_plot(self.data_recorder.database, flag_save=True, flag_show=True, filename='./res/wr_state.pdf')

    def local_plot(self, dict, flag_save=False, flag_show=True, filename='./', figsize=(8, 5)):
        # load data from data recorder
        keys_list = list(dict.keys())
        for name in keys_list:
            d = dict[name]
            globals()[name] = d

        fs_legend = 8
        def set_lim():
            ylim0 = plt.ylim()
            plt.ylim(ylim0[0], ylim0[1] + (ylim0[1] - ylim0[0]) * 0.15)

        # Profiles of states
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(311)
        plt.plot(t, theta, label=r'$x_1$')
        plt.plot(t, x_est[:, 0], label=r'$\hat x_{1}$', alpha=0.9, ls=(0, (5, 2)))
        plt.plot(t, y, label=r'$y$', linewidth=0.8, alpha=0.5)
        # plt.plot(t_lag, x_lag[:, 0], label=r'$\hat x_{1,\mathrm{FLKS}}$', ls=(0, (5, 2)), color=color2)
        plt.legend(ncol=4, fontsize=fs_legend, loc='upper left')
        plt.grid()
        set_lim()
        plt.ylabel('Roll angle [deg]')
        ax = fig.add_subplot(312)
        plt.plot(t, p, label=r'$x_2$')
        plt.plot(t, x_est[:, 1], label=r'$\hat x_{2}$', alpha=0.9, ls=(0, (5, 2)))
        # plt.plot(t_lag, x_lag[:, 1], label=r'$\hat x_{2, \mathrm{FLKS}}$', ls=(0, (5, 2)), alpha=1.0, color=color2)
        plt.legend(ncol=4, fontsize=fs_legend, loc='upper left')
        plt.grid()
        set_lim()
        plt.ylabel('Roll rate [deg/s]')
        ax = fig.add_subplot(313)
        plt.plot(t, Delta, label='$\Delta$')
        plt.plot(t, f_pre, label='$f_{\mathrm{GP}}$', alpha=0.9, ls=(0, (5, 2)))
        error_bar_plot(t, f_pre, np.sqrt(var_f) * 1.96, color='#350E62', alpha=0.2, label='95% Conf.')
        plt.legend(ncol=4, fontsize=fs_legend, loc='upper left')
        plt.grid()
        ylim0 = plt.ylim()
        plt.ylim(ylim0[0] + (ylim0[1] - ylim0[0]) * 0.25, ylim0[1] + (ylim0[1] - ylim0[0]) * -0.1)
        plt.ylabel('Roll acceleration\n [deg/s$^2$]')
        plt.xlabel('Time [s]')
        plt.subplots_adjust(left=0.17, bottom=0.10, right=None, top=0.93, wspace=0.3, hspace=None)
        save_show(flag_save, flag_show, filename, fig, dpi=None)

class MCsim():
    """Monte Carlo Simulation"""

    def __init__(self, tend=150, num_opt=1):
        self.tend = tend            # simulation time
        self.num_opt = num_opt      # number of gradient descent for hyperparameter optimization in a single RGPSSM iteration
        if self.num_opt == 0:
            self.log_path = './log/logMC.pkl'
        else:
            self.log_path = './log/logMC_hp.pkl'

    def run(self):
        self.Log = []
        var_seq = np.linspace(1, 10, 5)
        ls_seq = np.linspace(1, 5, 5)
        num_sim = 0
        for var in var_seq:
            for ls in ls_seq:
                obj = self.single_sim(var, ls)
                data = obj.data_recorder.database
                data['error'] = obj.error
                self.Log.append(data)
                num_sim += 1
                print(f'index: {num_sim}')

        save_pickle(self.Log, self.log_path)

    def single_sim(self, var, ls, flag_index=False, flag_plot=False, Qu=0., lr=1e-2):
        param = {}
        param['var'] = var
        param['ls'] = ls
        param['tend'] = self.tend
        param['num_opt_hp'] = self.num_opt
        param['Qu'] = Qu
        param['lr'] = lr
        obj = Learn(param)

        try:
            error = obj.run(flag_index)
        except RuntimeError:
            error = np.inf
        except ValueError:
            error = np.inf
        obj.error = error

        if flag_plot:
            obj.result_plot()

        return obj

    def result_plot(self, flag_save=False, flag_show=True, filename='./res/wr_hp.pdf'):
        self.Log = load_pickle(self.log_path)
        self.hp = RefHyperparam(tend=self.tend)
        self.hp.load()

        def hyperparam_plot(id):
            fs_legend = 11
            for ii in range(len(self.Log)):
                data = self.Log[ii]
                if id == 0:
                    hp = data['ls'][:, 0]
                    hp_ref = self.hp.ls.ravel()[0]
                    name = '$l_1$'
                    ax.set_xticks([])
                elif id == 1:
                    hp = data['ls'][:, 1]
                    hp_ref = self.hp.ls.ravel()[1]
                    name = '$l_2$'
                    ax.set_xticks([])
                elif id == 2:
                    hp = data['var'][:, 0]
                    hp_ref = self.hp.var.ravel()[0]
                    name = '$\sigma^2_1$'
                    ax.set_xlabel('Time [s]')
                ax.plot(data['t'], hp, alpha=0.6, linewidth=1)
            if id == 0:
                ax.plot(data['t'], np.ones_like(data['t'])*hp_ref, linewidth=2.3, alpha=0.8,
                        color='k', ls=(0, (5, 1)), label='reference value')
                plt.legend(loc='upper right', fontsize=fs_legend)
                plt.tick_params(axis='both', which='major', labelsize=fs_legend)
            else:
                ax.plot(data['t'], np.ones_like(data['t']) * hp_ref, linewidth=2.3, alpha=0.8,
                        color='k', ls=(0, (5, 1)))
            ax.set_ylabel(name)

        fig = plt.figure(figsize=(7, 4.5))
        for ii in range(3):
            ax = fig.add_subplot(3,1,ii+1)
            hyperparam_plot(ii)
        plt.subplots_adjust(left=0.12, bottom=0.12, right=0.88, top=0.97, wspace=0.3, hspace=0.08)
        save_show(flag_save, flag_show, filename, fig, dpi=None, flag_tight=False)

        for ii in range(len(self.Log)):
            d = self.Log[ii]
            l10, l20, var0 = d['ls'][0, 0], d['ls'][0, 1], d['var'][0, 0]
            l1, l2, var  = d['ls'][0, -1], d['ls'][-1, 1], d['var'][-1, 0]
            error = d['error']
            print(f'l10={l10}， l20={l20}， var0={var0}， l1={l1}， l2={l2}， var={var}, RMSE={error}')

    def compare_plot(self):
        self.Log = load_pickle('./log/logMC.pkl')
        self.LogHp = load_pickle('./log/logMC_hp.pkl')

        def error_plot(ax, Log, label):
            elist = [log['error'] for log in Log]
            ax.scatter(range(len(elist)), elist, label=label, s=5)
            return np.array(elist).mean()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        mean_error = error_plot(ax, self.Log, 'no hyperparamters adpatation')
        mean_error_hp = error_plot(ax, self.LogHp, 'with hyperparamters adpatation')
        plt.legend()
        plt.ylabel('RSME')

        print(f'mean error: {mean_error:.4f}')
        print(f'mean error with hyperparameters adaptation: {mean_error_hp:.4f}')

        improve = (mean_error - mean_error_hp) / mean_error
        print(f'improve = {improve*100}%')

if __name__ == '__main__':
    mc = MCsim(num_opt=1)
    # mc.run()
    mc.result_plot(flag_save=True)
    mc.compare_plot()
    plt.show()

