# -*- coding:utf-8 -*-
# @author  : Zheng Tengjie
# @time    : 2023/12/23 17:02
# @function: the dynamical system of wing rock.
# @version : V1

import numpy as np
from method import *


class WingRock():
    """Dynamical system of wing rock"""
    def __init__(self, delta_t=0.05):
        """
        delta_t : time_step [float]
        """

        self.dt = delta_t
        self.t = 0.0
        self.theta = 0.0
        self.p = 0.0
        self.L = 3 # control effective

        # parameters of uncertainty
        self.W = np.array([0.8, 0.2314, 0.6918, -0.6245, 0.0095, 0.0214]).reshape(-1, 1)

    def update(self, delta):
        """update the system by control input delta
        arguments:
        delta : control input [float]
        returns:
        theta : roll angle [float, deg]
        p : roll rate [float, deg/s]
        dot_p : roll acceleration [float, deg/s^2]
        Delta: uncertainty [float, deg/s^2]
        """

        self.delta = np.array(delta).item()
        X = RKM_4(np.array([self.theta, self.p]), self._dot_X, self.dt)
        self.theta, self.p = X[0], X[1]
        self.Delta = self.uncertainty(self.theta, self.p)
        self.dot_p = self.L * delta + self.Delta
        self.t += self.dt

        return self.theta, self.p, self.dot_p, self.Delta

    def uncertainty(self, theta, p):
        """calculate the uncertainty
        arguments:
        theta : roll angle [float, deg]
        p : roll rate [float, deg/s]
        returns:
        Delta : uncertainty [float, deg/s^2]
        """

        Phi = np.array([1.0, theta, p, np.abs(theta)*p, np.abs(p)*p, theta**3]).reshape(1, -1)
        Delta = Phi @ self.W

        return Delta.item()

    def _dot_X(self, X):
        """ODE of the system"""

        theta = X.ravel()[0]
        p = X.ravel()[1]
        Delta = self.uncertainty(theta, p)
        dot_X = np.array([p, self.L * self.delta + Delta])

        return dot_X


class Test():
    def __init__(self):
        np.random.seed(1)
        torch.manual_seed(1)
        self.obj = WingRock(delta_t=0.05)
        self.pid = PID(delta_t=self.obj.dt, Kp=4.0, Ki=2.0, Kd=2.0, flag_filter=True, filter_order=2, omega=1.7)
        self.t_last = -100
        self.sigma_noise = 0.2

        self.data_recorder = DataRecorder()

        self.t_learn = 0.
    def data_record_update(self):
        self.data_name = ['t', 'theta', 'theta_d', 'p', 'dot_p', 'delta', 'y', 'Delta']
        self.data_vector = [self.obj.t, self.obj.theta, self.theta_d, self.obj.p, self.obj.dot_p, self.delta, self.y, self.obj.Delta]
        self.data_recorder.data_add(self.data_name, self.data_vector)
    @timing
    def run(self):
        tend = 1000
        for ii in range(int(tend/self.obj.dt)):
            self.theta_d = np.sign(np.sin(2 * np.pi / 15 * self.obj.t)) * 1.5
            self.theta_d += np.sign(np.sin(2 * np.pi / 15 * self.obj.t + np.pi / 2)) * 1.5
            self.delta = self.pid.update(self.theta_d, self.obj.theta)
            self.obj.update(delta=self.delta)
            self.y = self.obj.theta + np.random.randn() * self.sigma_noise

            self.data_record_update()
            if ii % 100 == 0:
                print(f'ii = {ii}')

    def result_plot(self):
        # load data from data recorder
        dict = self.data_recorder.database
        keys_list = list(dict.keys())
        for name in keys_list:
            d = dict[name]
            globals()[name] = d

        # profiles of states
        fig = plt.figure(figsize=(17, 4))
        ax = fig.add_subplot(141)
        plt.plot(t, theta_d, label=r'$\theta_d$')
        plt.plot(t, theta, label=r'$\theta$')
        plt.plot(t, y, label=r'$y$')
        plt.legend()
        plt.xlabel('Time [s]')
        ax = fig.add_subplot(142)
        plt.plot(t, p, label=r'$p$')
        plt.legend()
        plt.xlabel('Time [s]')
        ax = fig.add_subplot(143)
        plt.plot(t, dot_p, label='$\dot p$')
        plt.plot(t, Delta, label='$\Delta$')
        plt.legend()
        plt.xlabel('Time [s]')
        ax = fig.add_subplot(144)
        plt.plot(t, delta, label='$\delta$')
        plt.legend()
        plt.xlabel('Time [s]')
        plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.3, hspace=None)

if __name__ == '__main__':
    test = Test()
    test.run()
    save_pickle(test, './log/wr.pkl')
    test.result_plot()
    print('finish!')
    plt.show()