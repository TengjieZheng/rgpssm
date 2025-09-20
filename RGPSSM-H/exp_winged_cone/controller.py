from typing import Tuple, Optional, List, Union
from types import SimpleNamespace

from torch import Tensor
from numpy import ndarray

import numpy as np

from vehicle.geometry import get_aerodynamic_angles
from vehicle.utils import RKM_4
from vehicle.earth import Earth

class Controller():
    def __init__(self, wc, fun_C):
        """
        Args:
            wc:     winged cone module
            fun_C:  function to evalute moment coefficients (alpha, beta, omg, delta) -> C
        """
        self.fun_C = fun_C
        self.wc = wc
        self.dt = self.wc.delta_t

        self.x_ref = np.zeros((6, 1))           # state of the reference model
        self.x2_f = np.zeros((3, 1))            # filtered angular rate command
        self.xi_ref, self.omg_ref = 1., 10.     # damping ratio and natural frequency for the reference model
        self.T_filter2 = 1 / self.omg_ref / 3   # filtering time constant for the angular rate command
        self.K1 = np.eye(3) * 3.                # gain for the angle subsystem
        self.K2 = self.K1 * 0.5                 # gain for the angular rate subsystem

        self.num_update = 0

    def update(self, state, x1_c, fun_C=None):
        """
        Args:
            x1_c: attitude angle command [roll_c, pitch_c, yaw_c]
        Returns:
            u: [delta_a, delta_e, delta_r]
        """
        self.fun_C = fun_C if fun_C is not None else self.fun_C
        roll, pitch, yaw, alpha, beta, phi_v, omg, h, V = self.read_state(state)

        if self.num_update == 0:
            self.x_ref[:3, 0] = x1_c.ravel()

        self.angle_subsys_update(roll, pitch, yaw, x1_c)    # Angle subsystem: get angular rate command x2_c
        self.angular_rate_subsys_update(omg, state)         # Angular rate subsystem: get control input u

        self.num_update += 1

        return self.u

    def angle_subsys_update(self, roll, pitch, yaw, x1_c):
        """Get angular rate command x2_c"""
        x1 = np.array([roll, pitch, yaw]).reshape(-1, 1)
        x1_c = np.array(x1_c).reshape(-1, 1)

        def ref_model(x):
            x1_f, dot_x1_f = x[0:3, :], x[3:6, :]
            ddot_x1_f = -2 * self.xi_ref * self.omg_ref * dot_x1_f + self.omg_ref ** 2 * (x1_c - x1_f)
            return np.vstack((dot_x1_f, ddot_x1_f))

        self.x_ref = RKM_4(self.x_ref, ref_model, self.dt)
        x1_f, dot_x1_f = self.x_ref[0:3, :], self.x_ref[3:6, :]

        ## Command: x2_c
        g = np.array([[1., np.tan(pitch) * np.sin(roll), np.tan(pitch) * np.cos(roll)],
                       [0., np.cos(roll), -np.sin(roll)],
                       [0., np.sin(roll) / np.cos(pitch), np.cos(roll) / np.cos(pitch)]])

        dot_x1_d = self.K1 @ (x1_f - x1) + dot_x1_f
        self.x2_c = np.linalg.solve(g, dot_x1_d)

    def angular_rate_subsys_update(self, omg, state):
        """Get control input u"""
        ## Filter
        def filter(x):
            return 1 / self.T_filter2 * (self.x2_c - x)

        dot_x2_f = filter(self.x2_f)
        self.x2_f = RKM_4(self.x2_f, filter, self.dt)

        ## Command: u
        e2 = self.x2_f - omg
        F, G = self.get_f_g(state)
        dot_omg_d = self.K2 @ e2 + dot_x2_f
        self.u = np.linalg.solve(G, dot_omg_d - F)
        self.magnitute_clip()  # clip delta


    def get_f_g(self, state):
        """Get F(s, 0) and control effect G
        """
        roll, pitch, yaw, alpha, beta, phi_v, omg, h, V = self.read_state(state)

        # get zero-order term
        D_delta = -2. / 57.3
        s = [alpha, beta, omg, h, V]
        f = self.F(s, np.zeros((3, 1))) # (3, 1)

        # get control effect
        F_dist = []
        for i in range(3):
            delta_dist = np.zeros((3, 1))
            delta_dist[i, 0] = D_delta
            f_dist = self.F(s, delta_dist)   # (3, 1)
            F_dist.append(f_dist)
        F_dist = np.concatenate(F_dist, axis=-1)
        g = (F_dist - f) / D_delta      # (3, 3)

        return f, g

    def F(self, s: List, delta: ndarray):
        """
        Args:
            s: alpha, beta, omg, h, V
            delta: (3,)
        Returns:
        """
        alpha, beta, omg, h, V = s
        dp = Earth.dynamic_pressure(h, V)

        # Get areodynamic moment coefficients
        C = self.fun_C(alpha, beta, omg, delta, h, V)
        C = np.array(C).reshape(-1, 1)

        # calculate moment and angular acceleration
        M_i = -np.cross(omg, self.wc.I_mat @ omg, axis=0)                       # (3, 1)
        len_vec = np.array([self.wc.b, self.wc.c, self.wc.b]).reshape(-1, 1)
        M_ad = dp * self.wc.S_ref * len_vec * C
        M = M_i + M_ad
        dot_omg = np.linalg.solve(self.wc.I_mat, M)

        return dot_omg

    def read_state(self, state):
        z, V, gamma, chi, roll, pitch, yaw, p, q, r = (
            state['z'], state['V'], state['gamma'], state['chi'], state['roll'], state['pitch'], state['yaw'],
            state['p'], state['q'], state['r'])

        alpha, beta, phi_v = get_aerodynamic_angles(gamma, chi, roll, pitch, yaw)
        omg = np.array([p, q, r]).reshape(-1, 1)
        h = -z

        return roll, pitch, yaw, alpha, beta, phi_v, omg, h, V

    def magnitute_clip(self):
        """Clip magnitute of control surface deflection"""
        self.delta_a, self.delta_e, self.delta_r = self.u
        self.delta_a = np.clip(self.delta_a, -20. / 57.3, 20. / 57.3)
        self.delta_e = np.clip(self.delta_e, -20. / 57.3, 20. / 57.3)
        self.delta_r = np.clip(self.delta_r, -20. / 57.3, 20. / 57.3)
        self.u = np.array([self.delta_a, self.delta_e, self.delta_r]).reshape(-1, 1)