import matplotlib.pyplot as plt
from monte_carlo import *

class SimCompare():
    """Compare the learning result with and without GP hyperparameters online adaptation"""
    def __init__(self, tend=50, var=10, ls=5):
        self.log_path = './log/log_single_sim.pkl'
        self.tend = tend    # simulation time
        self.var = var      # initial GP kernel hyperparameters: signal variance
        self.ls = ls        # initial GP kernel hyperparameters: length scale

    def run(self):
        # simulation without hyperparameter optimization
        sim = MCsim(tend=self.tend, num_opt=0)
        obj = sim.single_sim(self.var, self.ls, flag_index=True, flag_plot=False)

        # simulation with hyperparameter optimization
        sim_hp = MCsim(tend=self.tend, num_opt=1)
        obj_hp = sim_hp.single_sim(self.var, self.ls, flag_index=True, flag_plot=False)

        # Save data
        d = obj.data_recorder.database
        d_hp = obj_hp.data_recorder.database
        self.save(d, d_hp)

    def save(self, d, d_hp):
        save_pickle([d, d_hp], file_name=self.log_path)

    def load(self):
        d, d_hp = load_pickle(self.log_path)
        return d, d_hp

    def result_plot(self, flag_save=False, flag_show=True, filename='./res/wr_state.pdf'):

        def set_lim():
            ylim0 = plt.ylim()
            plt.ylim(ylim0[0], ylim0[1] + (ylim0[1] - ylim0[0]) * 0.15)

        d, d_hp = self.load()

        t, theta, y, p, Delta = d['t'], d['theta'], d['y'], d['p'], d['Delta']
        x_est, f_pre, var_f = d['x_est'], d['f_pre'], d['var_f']
        x_est_hp, f_pre_hp, var_f_hp = d_hp['x_est'], d_hp['f_pre'], d_hp['var_f']

        # Profiles of states
        fs_legend = 10
        fig = plt.figure(figsize=(7, 5.))
        ax = fig.add_subplot(311)
        plt.plot(t, theta, label=r'$z_1$', zorder=0)
        plt.plot(t, x_est[:, 0], label=r'$\hat z_{1}$', alpha=0.9, ls=(0, (8, 3)), zorder=1)
        plt.plot(t, x_est_hp[:, 0], label=r'$\hat z_{1, \mathrm{adapt}}$', alpha=0.9, ls=(0, (5, 3)), zorder=2)
        plt.scatter(t, y, label=r'measurement', linewidth=0.8, alpha=0.4, color='r', s=1.5, zorder=0)
        plt.legend(ncol=4, fontsize=fs_legend, loc='upper left')
        plt.tick_params(axis='both', which='major', labelsize=fs_legend)
        plt.xticks([])
        set_lim()
        plt.ylabel('Roll angle [deg]')

        ax = fig.add_subplot(312)
        plt.plot(t, p, label=r'$z_2$')
        plt.plot(t, x_est[:, 1], label=r'$\hat z_{2}$', alpha=0.9, ls=(0, (8, 3)))
        plt.plot(t, x_est_hp[:, 1], label=r'$\hat z_{2, \mathrm{adapt}}$', alpha=0.9, ls=(0, (5, 3)))
        plt.legend(ncol=4, fontsize=fs_legend, loc='upper left')
        plt.tick_params(axis='both', which='major', labelsize=fs_legend)
        plt.xticks([])
        set_lim()
        plt.ylabel('Roll rate [deg/s]')

        ax = fig.add_subplot(313)
        plt.plot(t, Delta, label='$\Delta$')
        line =plt.plot(t, f_pre, label='$f_{\mathrm{GP}}$', alpha=0.9, ls=(0, (8, 3)))
        line_hp =plt.plot(t, f_pre_hp, label='$f_{\mathrm{GP}, \mathrm{adapt}}$', alpha=0.9, ls=(0, (5, 3)))
        line_color = line[0].get_color()
        line_hp_color = line_hp[0].get_color()
        error_bar_plot(t, f_pre, np.sqrt(var_f) * 1.96, color=line_color, alpha=0.4, label=None)
        error_bar_plot(t, f_pre_hp, np.sqrt(var_f_hp) * 1.96, color=line_hp_color, alpha=0.3, label=None)
        plt.legend(ncol=4, fontsize=fs_legend, loc='upper left')
        plt.tick_params(axis='both', which='major', labelsize=fs_legend)
        ylim0 = plt.ylim()
        plt.ylim(ylim0[0] + (ylim0[1] - ylim0[0]) * 0.25, ylim0[1] + (ylim0[1] - ylim0[0]) * -0.1)
        plt.ylabel('Uncertainty [deg/s$^2$]')
        plt.xlabel('Time [s]')

        plt.subplots_adjust(left=0.12, bottom=0.1, right=0.88, top=0.97, wspace=0.3, hspace=0.05)
        save_show(flag_save, flag_show, filename, fig, dpi=None, flag_tight=False)

class InducingPointEvolution():
    """Show the evolution of inducing points during learning process"""

    def __init__(self, var=10, ls=5):
        self.log_path = './log/log_inducing_point.pkl'
        self.var = var      # initial GP kernel hyperparameters: signal variance
        self.ls = ls        # initial GP kernel hyperparameters: length scale

    def run(self):
        tend_list = [1., 4., 7., 10., 13.]
        obj_list = []
        for tend in tend_list:
            obj = self.single_sim(tend)
            obj_list.append(obj)
        self.save(obj_list)

    def single_sim(self, tend):
        sim = MCsim(tend=tend, num_opt=0)
        obj = sim.single_sim(self.var, self.ls, flag_index=True, flag_plot=False)

        return obj

    def single_plot(self, log, ax, flag_lenged=True):
        """Inducing points distribution for single simulation"""

        data = log.data_recorder.database
        theta, p = data['theta'], data['p']

        theta_arr = np.linspace(-3.5, 3.5, 30)
        p_arr = np.linspace(-3.5, 3.5, 30)
        theta_mg, p_mg = np.meshgrid(theta_arr, p_arr)

        def Delta_true(x):
            return log.obj.wr.uncertainty(x.ravel()[0], x.ravel()[1])

        f_true_mg = meshgrid_cal([theta_mg, p_mg], Delta_true, num_output=1)

        def Delta_pre(x):
            f, var = log.gpssm.GP_predict(x.reshape(1, -1))
            f, var = Torch2Np(f), Torch2Np(var)

            return Np2Num(f), Np2Num(var)

        f_pre_mg, Var_mg = meshgrid_cal([theta_mg, p_mg], Delta_pre, num_output=2)

        cmap = 'plasma'
        cmap = 'viridis'

        vmin = 0.
        vmax = 7.5
        e_mg = np.abs(f_true_mg - f_pre_mg)
        surf = ax.contourf(theta_mg, p_mg, e_mg, 20, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
        BV = Torch2Np(log.gpssm.Z)
        ax.scatter(theta, p, label='sample', marker='o', facecolor='none', edgecolors='w', s=20., alpha=0.8,
                    linewidths=0.50)
        ax.scatter(BV[:, 0], BV[:, 1], label=r'inducing point', marker='D', s=30, alpha=0.9, facecolors='none',
                    edgecolors='y', linewidths=1.8)
        if flag_lenged:
            plt.legend(loc='lower left')
        ax.set_xlabel(r'$\theta$ [deg]')
        ax.set_ylabel(r'$p$ [deg/s]')
        tend = log.param['tend']
        plt.title(f'$t = {tend}$s: {log.gpssm.Z.shape[0]} points' )
        plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.3, hspace=None)

        return surf

    def result_plot(self, flag_save=False, flag_show=True, filename='./res/wr_inducing_point.pdf'):
        obj_list = self.load()

        fig = plt.figure(figsize=(19, 3.5))
        surf_list = []
        for ii, obj in enumerate(obj_list):
            ax = fig.add_subplot(1, len(obj_list), ii + 1)
            if ii == 0:
                surf = self.single_plot(obj, ax, flag_lenged=True)
            else:
                surf = self.single_plot(obj, ax, flag_lenged=False)
            surf_list.append(surf)

        fig.get_axes()[-1].text(2.8, 3.8, 'prediction\n     error', size=12)

        # Add a colorbar using the first contourf object for reference
        cbar = fig.colorbar(surf_list[0], ax=fig.get_axes(), shrink=0.85, aspect=30, fraction=0.1, pad=0.5)
        cbar.ax.tick_params(labelsize=10)
        plt.subplots_adjust(left=0.05, bottom=0.18, right=0.815, top=0.85, wspace=0.3, hspace=None)

        save_show(flag_save=flag_save, flag_show=flag_show, filename=filename, fig=fig, dpi=300)

    def save(self, obj_list):
        save_pickle(obj_list, self.log_path)

    def load(self):
        return load_pickle(self.log_path)

if __name__ == '__main__':
    test = SimCompare(tend=50)
    test.run()
    test.result_plot(flag_save=True)

    test = InducingPointEvolution()
    test.run()
    test.result_plot(flag_save=True)
    plt.show()

