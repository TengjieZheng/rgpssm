import copy

import numpy as np
import matplotlib.pyplot as plt

from .utils import RKM_4, Np2Num
from .aerodynamics import Aerodynamics, Thrust
from .earth import Earth
from .geometry import *

class Winged_cone():
    """6DOF winged-cone model
    Coordinate System: European-American System (x-y-z: front-right-down)
    Units: Imperial Units
    """
    def __init__(self, delta_t=0.01, flag_attitude=True, pull=None):
        """
        delta_t : time step [float]
        flag_attitude : whether to simulate attitude motion only [bool]
        pull: pull coefficient [dict]
        """
        self.flag_attitude = flag_attitude # whether to simulate attitude motion only

        self.earth = Earth()
        self.ad = Aerodynamics(pull)

        # Time parameters
        self.delta_t = delta_t
        self.t = 0.

        # Mass parameters
        self.m = 9324.285   # mass
        self.I_xx = 6.4e5   # inertia moment
        self.I_yy = 7e6
        self.I_zz = 9e6
        self.I_xz = 1e5

        # Geometry parameters
        self.S_ref = 3603   # reference area
        self.c = 80         # mean aerodynamic chord
        self.b = 60         # span

        # Engine parameters
        self.omg_engine, self.xi_engine = 4., 1.                # natural frequency, damping ratio for the engine dynamics

        # State initialization (nominal cruise condition)
        position = np.array([0., 0., -110000.]).reshape(-1, 1)  # x, y, z [m]
        velocity = np.array([15060 * 3. / 3, 0., 0.]).reshape(-1, 1)    # V [m/s], flight_path_angle [rad], heading_angle [rad]
        angle = np.array([0., 0.0315, 0.]).reshape(-1, 1)       # roll, pitch, yaw [rad]
        rate = np.array([0., 0., 0.]).reshape(-1, 1)            # omega_x/p, omega_y/q, omega_z/r [rad/s]
        throttle_state = np.array([0., 0.]).reshape(-1, 1)      # throttle angle, throttle angular rate [rad, rad/s]
        self.x = np.vstack((position, velocity, angle, rate, throttle_state))
        self.x = copy.deepcopy(self.x)

        # Control inputs
        self.delta_a, self.delta_e, self.delta_r = 0., 0., 0.   # right elevon, left elevon, rudder [rad]
        self.thr_c = 0.183                                      # throttle angle command [rad]
        self.thr_c = 0.

        # Guidance parameters
        self.K_PN = 2.2
        self.roll_c, self.pitch_c, self.yaw_c = 0., 0.0315, 0.

        # Control parameters
        self.x_ref = np.zeros((6, 1))
        self.x2_d = np.zeros((3, 1))
        self.xi_ref, self.omg_ref = 1., 1.
        self.T_filter2 = 1 / self.omg_ref / 3
        self.K1 = np.eye(3) * 5.
        self.K2 = self.K1 * 5

        # Noise parameters
        self.sigma_p = 30.                              # noise standard deviation for position
        self.sigma_v = 50.                              # noise standard deviation for velocity
        self.sigma_angle = 0.1 / 57.3                   # noise standard deviation for angle
        self.sigma_rate = 0.05/ 57.3 * np.ones(3)       # noise standard deviation for rate

        # Initialization
        self.num_update = 0
        self.target_update()
        self.controller_update()
        self.update_C()
        self.senser_update()

    def update(self, u=None, update_tgt=True):
        """Update the state of vehicle"""

        if update_tgt: self.target_update()                     # attitude angle command update
        if u is not None:
            self.delta_a, self.delta_e, self.delta_r = np.array(u).ravel()
        else:
            self.controller_update()                            # attitude controller update
        self.update_C()                                         # update aerodynamics cofficients
        self.x = RKM_4(self.x, self.dynamics, self.delta_t)     # state update
        self.senser_update()                                    # senser update
        self.t += self.delta_t

        self.num_update += 1

    def target_update(self):
        """Update attitude angle command"""
        x, y, z, V, gamma, chi, roll, pitch, yaw, p, q, r, thr, thr_rate = self.x.ravel()

        self.alpha_c = np.sin((1 + self.t / 50) * self.t/2.) * 5 / 57.3 + 3 / 57.3
        self.phi_v_c = np.sin((1 + self.t / 50) * self.t/2.) * 10 / 57.3
        self.beta_c = np.sin((1 + self.t / 50) * self.t/2.) * 2. / 57.3 - 0 /57.3
        self.roll_c, self.pitch_c, self.yaw_c = get_attidude_angles(gamma, chi, self.alpha_c, self.beta_c, self.phi_v_c)
        self.angle_c = np.array([self.roll_c, self.pitch_c, self.yaw_c]).reshape(-1, 1)

        if self.num_update == 0:
            self.x_ref[:3, 0] = np.array([self.roll_c, self.pitch_c, self.yaw_c])
            self.x[-8:-5, :] = self.x_ref[:3, :]

    def controller_update(self):
        """Attitude controller update"""

        # Read states
        x, y, z, V, gamma, chi, roll, pitch, yaw, p, q, r, thr, thr_rate = self.x.ravel()
        alpha, beta, phi_v = get_aerodynamic_angles(gamma, chi, roll, pitch, yaw)
        omg = np.array([p, q, r]).reshape(-1, 1)
        h = -z

        # Subsystem 1: Angle
        ## Reference model
        x1 = np.array([roll, pitch, yaw]).reshape(-1, 1)
        x1_c = np.array([self.roll_c, self.pitch_c, self.yaw_c]).reshape(-1, 1)
        def ref_model(x):
            x1_d, dot_x1_d = x[0:3, :], x[3:6, :]
            ddot_x1_d = -2 * self.xi_ref * self.omg_ref * dot_x1_d + self.omg_ref ** 2 * (x1_c - x1_d)
            return np.vstack((dot_x1_d, ddot_x1_d))
        self.x_ref = RKM_4(self.x_ref, ref_model, self.delta_t)
        x1_d, dot_x1_d = self.x_ref[0:3, :], self.x_ref[3:6, :]

        ## Command: x2_c
        g1 = np.array([[1., np.tan(pitch)*np.sin(roll), np.tan(pitch)*np.cos(roll)],
                       [0., np.cos(roll), -np.sin(roll)],
                       [0., np.sin(roll)/np.cos(pitch), np.cos(roll)/np.cos(pitch)]])
        inv_g1 = np.linalg.inv(g1)
        self.x2_c = inv_g1 @ (self.K1 @ (x1_d - x1) + dot_x1_d)

        # Subsystem 2: Angular rate
        ## Filter
        def filter(x):
            return 1 / self.T_filter2 * (self.x2_c - x)
        dot_x2_d = filter(self.x2_d)
        self.x2_d = RKM_4(self.x2_d, filter, self.delta_t)

        ## Command: u
        x2 = np.array([p, q, r]).reshape(-1, 1)
        fM, gM = self.ad.Calculate_fM_GM(h, V, alpha, beta, p, q, r)
        I_mat = self.I_mat
        M_i = -np.cross(omg, I_mat @ omg, axis=0)  # inertial moment
        f2 = M_i + fM
        f2 = np.linalg.solve(I_mat, f2)
        g2 = np.linalg.solve(I_mat, gM)

        inv_g2 = np.linalg.inv(g2)
        e2 = self.x2_d - x2
        u = inv_g2 @ (dot_x2_d - f2 + self.K2 @ e2)
        self.delta_a, self.delta_e, self.delta_r = u.ravel()

        self.delta_a = np.clip(self.delta_a, -20./57.3, 20./57.3)
        self.delta_e = np.clip(self.delta_e, -20./57.3, 20./57.3)
        self.delta_r = np.clip(self.delta_r, -20./57.3, 20./57.3)

    def dynamics(self, x):
        """Dynamics of vehicle
        Arguments:
            x : state [ndarray]
        Returns:
            dot_x : state rate [ndarray]
        """

        # Read states and calculate aerodynamic angles
        x, y, z, V, gamma, chi, roll, pitch, yaw, p, q, r, thr, thr_rate = x.ravel()
        alpha, beta, phi_v = get_aerodynamic_angles(gamma, chi, roll, pitch, yaw)
        omg = np.array([p, q, r]).reshape(-1, 1)

        # Calculate force and moment
        h = -z # height
        g = self.get_g(h)
        L, D, Y, M_roll, M_pitch, M_yaw = self.ad.get_force(h, V, alpha, beta, self.delta_a, self.delta_e, self.delta_r, p, q, r, flag_pull=True)
        T = Thrust(h, V, thr)

        # Dynamic equations
        if self.flag_attitude:
            dot_V = 0.
            dot_chi = 0.
            dot_gamma = 0.
        else:
            dot_V = -g * np.sin(gamma) + (T * np.cos(beta) * np.cos(alpha) - D) / self.m
            dot_chi = (T * (np.sin(phi_v) * np.sin(alpha) - np.cos(phi_v) * np.sin(beta) * np.cos(alpha))
                       + Y * np.cos(phi_v) + L * np.sin(phi_v)) / self.m / V / np.cos(gamma)
            dot_gamma = (g * np.cos(gamma) + (
                        T * (-np.sin(phi_v) * np.sin(beta) * np.cos(alpha) - np.cos(phi_v) * np.sin(alpha))
                        + Y * np.sin(phi_v) - L * np.cos(phi_v)) / self.m) / (-V)

        M_ad = np.array([M_roll, M_pitch, M_yaw]).reshape(-1, 1)    # aerodynamic moment
        I_mat = self.I_mat
        M_i = -np.cross(omg, I_mat @ omg, axis=0)                   # inertial moment

        dot_omg = np.linalg.solve(I_mat, M_i + M_ad)
        dot_p, dot_q, dot_r = dot_omg.ravel()

        # Kinematic equations
        dot_x = V * np.cos(gamma) * np.cos(chi)
        dot_y = V * np.cos(gamma) * np.sin(chi)
        dot_z = -V * np.sin(gamma)

        dot_roll = p + np.tan(pitch) * (q*np.sin(roll) + r*np.cos(roll))
        dot_pitch = q*np.cos(roll) - r*np.sin(roll)
        dot_yaw = (q*np.sin(roll) + r*np.cos(roll)) / np.cos(pitch)

        # Engine dynamics
        dot_thr = thr_rate
        dot_thr_rate = -2*self.xi_engine*self.omg_engine * thr_rate + self.omg_engine**2 * (self.thr_c - thr)

        # State rate
        dot_x = np.array([dot_x, dot_y, dot_z, dot_V, dot_gamma, dot_chi, dot_roll, dot_pitch, dot_yaw, dot_p, dot_q, dot_r, dot_thr, dot_thr_rate]).reshape(-1, 1)

        return dot_x

    def senser_update(self):
        """Update sensor state, which is a dictionary consisting of state and measurements"""

        x, y, z, V, gamma, chi, roll, pitch, yaw, p, q, r, thr, thr_rate = self.x.ravel()
        alpha, beta, phi_v = get_aerodynamic_angles(gamma, chi, roll, pitch, yaw)
        
        x_meas, y_meas, z_meas = x + np.random.randn() * self.sigma_p, y + np.random.randn() * self.sigma_p, z + np.random.randn() * self.sigma_p
        V_meas, gamma_meas, chi_meas = V + np.random.randn() * self.sigma_v, gamma + np.random.randn() * self.sigma_angle, chi + np.random.randn() * self.sigma_angle
        roll_meas, pitch_meas, yaw_meas = roll + np.random.randn() * self.sigma_angle, pitch + np.random.randn() * self.sigma_angle, yaw + np.random.randn() * self.sigma_angle
        alpha_meas, beta_meas, phi_v_meas = alpha + np.random.randn() * self.sigma_angle, beta + np.random.randn() * self.sigma_angle, phi_v + np.random.randn() * self.sigma_angle
        p_meas, q_meas, r_meas = p + np.random.randn() * self.sigma_rate[0], q + np.random.randn() * self.sigma_rate[1], r + np.random.randn() * self.sigma_rate[2]



        self.state = {
            't': self.t,
            'x': x,
            'y': y,
            'z': z,
            'V': V,
            'gamma': gamma,
            'chi': chi,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'p': p,
            'q': q,
            'r': r,
            'omg': np.array([p, q, r]).reshape(-1, 1),
            'thr': thr,
            'thr_rate': thr_rate,
            'alpha': alpha,
            'beta': beta,
            'phi_v': phi_v,
            'delta_a': self.delta_a,
            'delta_e': self.delta_e,
            'delta_r': self.delta_r,
            'roll_c': self.roll_c,
            'pitch_c': self.pitch_c,
            'yaw_c': self.yaw_c,
            'x2_c': self.x2_c,
            'x2_d': self.x2_d,
            'x1_d': self.x_ref[0:3, :],
            'Q': self.Q,
            'Ma': V / Earth.V_sound(-z),
            'rho' : Earth.rho(-z),
            'Vs' : Earth.V_sound(-z),
            'alpha_c': self.alpha_c,
            'beta_c': self.beta_c,
            'phi_v_c': self.phi_v_c,
            'C_force': self.C_force,
            'C_moment': self.C_moment,
            'C_force_0': self.C_force_0,
            'C_moment_0': self.C_moment_0,
            'x_meas': x_meas,
            'y_meas': y_meas,
            'z_meas': z_meas,
            'V_meas': V_meas,
            'gamma_meas': gamma_meas,
            'chi_meas': chi_meas,
            'roll_meas': roll_meas,
            'pitch_meas': pitch_meas,
            'yaw_meas': yaw_meas,
            'alpha_meas': Np2Num(alpha_meas),
            'beta_meas': Np2Num(beta_meas),
            'phi_v_meas': Np2Num(phi_v_meas),
            'p_meas': p_meas,
            'q_meas': q_meas,
            'r_meas': r_meas
        }

        return self.state

    @property
    def I_mat(self):
        "inertia matrix"
        I = np.zeros((3, 3))
        I[0, 0] = self.I_xx
        I[1, 1] = self.I_yy
        I[2, 2] = self.I_zz
        I[0, 2] = -self.I_xz
        I[2, 0] = -self.I_xz
        return I

    @property
    def v_vec(self):
        """Get velocity vector"""
        x, y, z, V, gamma, chi, roll, pitch, yaw, p, q, r, thr, thr_rate = self.x.ravel()

        velocity_Su = np.array([V * np.cos(gamma) * np.cos(chi),
                                V * np.cos(gamma) * np.sin(chi),
                                -V * np.sin(gamma)]).reshape(-1, 1)

        return velocity_Su

    @property
    def Q(self):
        """Dynamic pressure"""
        return 0.5 * Earth.rho(-self.x[2, 0]) * self.x[3, 0]**2

    def get_g(self, h):
        """Get gravity acceleration"""
        g = self.earth.mu / (self.earth.Re + h)**2
        return Np2Num(g)

    def update_C(self):
        """Update the areodynamic moment coefficient"""
        x, y, z, V, gamma, chi, roll, pitch, yaw, p, q, r, thr, thr_rate = self.x.ravel()
        alpha, beta, phi_v = get_aerodynamic_angles(gamma, chi, roll, pitch, yaw)
        CL, CD, CY, Cl, Cm, Cn = self.ad.get_C(alpha, beta, self.delta_a, self.delta_e, self.delta_r, p, q, r, V, -z, flag_pull=True)
        self.C_force = np.array([CL, CD, CY])       # true value
        self.C_moment = np.array([Cl, Cm, Cn])      # true value

        CL, CD, CY, Cl, Cm, Cn = self.ad.get_C(alpha, beta, self.delta_a, self.delta_e, self.delta_r, p, q, r, V, -z, flag_pull=False)
        self.C_force_0 = np.array([CL, CD, CY])     # baseline value
        self.C_moment_0 = np.array([Cl, Cm, Cn])    # baseline value

