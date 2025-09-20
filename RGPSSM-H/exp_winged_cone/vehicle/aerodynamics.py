import copy

import numpy as np

from .utils import Np2Num
from .earth import Earth

flag_damping = 1

def fun_CL(Ma, alpha, delta_a, delta_e):
    CL_a = -8.19e-02 + Ma * 4.70e-02 + alpha * 1.86e-02 - (alpha * Ma) * 4.73e-04 \
           - (Ma ** 2) * 9.19e-03 - (alpha ** 2) * 1.52e-04 + ((alpha * Ma) ** 2) * 5.99e-07 \
           + (Ma ** 3) * 7.74e-04 + (alpha ** 3) * 4.08e-06 - (Ma ** 4) * 2.93e-05 \
           - (alpha ** 4) * 3.91e-07 + (Ma ** 5) * 4.12e-07 + (alpha ** 5) * 1.30e-08

    def fun_CL_delta_a(Ma, alpha, delta_a):
        return -1.45e-05 + alpha * 1.01e-04 + Ma * 7.10e-06 - delta_a * 4.14e-04 \
               - (alpha * delta_a) * 3.51e-06 + (alpha * Ma) * 4.70e-06 + (Ma * delta_a) * 8.72e-06 - (
                       (alpha * Ma) * delta_a) * 1.70e-07

    CL_delta_a = fun_CL_delta_a(Ma, alpha, delta_a)
    CL_delta_e = fun_CL_delta_a(Ma, alpha, delta_e)
    CL = CL_a + CL_delta_a + CL_delta_e

    return CL


def fun_CD(Ma, alpha, delta_a, delta_e, delta_r):

    if alpha < 0:
       alpha = -alpha

    CD_a = 8.717e-02 - Ma * 3.307e-02 + alpha * 3.179e-03 - (alpha * Ma) * 1.250e-04 \
           + (Ma ** 2) * 5.036e-03 - (alpha ** 2) * 1.100e-03 \
           + ((alpha * Ma) ** 2) * 1.405e-07 - (Ma ** 3) * 3.658e-04 + (alpha ** 3) * 3.175e-04 \
           + (Ma ** 4) * 1.274e-05 - (alpha ** 4) * 2.985e-05 \
           - (Ma ** 5) * 1.705e-07 + (alpha ** 5) * 9.766e-07

    def fun_CD_delta_a(Ma, alpha, delta_a):
        return 4.5548e-04 + alpha * 2.5411e-05 + Ma * (-1.1436e-04) + delta_a * (-3.6417e-05) \
               + ((alpha * Ma) * delta_a) * (-5.3015e-07) + (alpha ** 2) * 3.2187e-06 \
               + (Ma ** 2) * 3.0140e-06 + (delta_a ** 2) * 6.9629e-06 \
               + (((alpha * Ma) * delta_a) ** 2) * 2.1026e-012

    CD_delta_a = fun_CD_delta_a(Ma, alpha, delta_a)
    CD_delta_e = fun_CD_delta_a(Ma, alpha, delta_e)
    CD_delta_r = 7.50e-04 - alpha * 2.2900e-05 - Ma * 9.6900e-05 - delta_r * 1.8300e-06 \
                 + ((alpha * Ma) * delta_r) * 9.13e-09 + (alpha ** 2) * 8.7600e-07 \
                 + (Ma ** 2) * 2.7000e-06 + (delta_r ** 2) * 1.9701e-06 \
                 - (((alpha * Ma) * delta_r) ** 2) * 1.7702e-11

    CD = CD_a + CD_delta_a + CD_delta_e + CD_delta_r

    return CD


def fun_CY(Ma, alpha, beta, delta_a, delta_e, delta_r):
    CY_beta = - 2.9253e-001 + alpha * 2.803e-003 + (alpha * Ma) * (-2.8943e-004) \
              + (Ma ** 2) * 5.4822e-002 + (alpha ** 2) * 7.3535e-004 + ((alpha * Ma) ** 2) * (-4.6490e-009) \
              + ((alpha ** 2) * Ma) ** 2 * (-2.0675e-008) + ((alpha * Ma) ** 2) * 4.6205e-006 \
              + ((alpha ** 2) * Ma ** 2) ** 2 * 2.6144e-011 + (Ma ** 3) * (-4.3203e-003) \
              + (alpha ** 3) * (-3.7405e-004) + (Ma ** 4) * 1.5495e-004 \
              + (alpha ** 4) * 2.8183e-005 + (Ma ** 5) * (-2.0829e-006) + (alpha ** 5) * (-5.2083e-007)

    def fun_CY_delta_a(Ma, alpha, delta_a):
        return -1.02e-06 - alpha * 1.12e-07 + Ma * 4.48e-07 + delta_a * 2.27e-07 \
               + ((alpha * Ma * delta_a) * 4.11e-09 + (alpha ** 2) * 2.82e-09 - (Ma ** 2) * 2.36e-08 \
                  - (delta_a ** 2) * 5.04e-08 + ((alpha * Ma * delta_a) ** 2) * 4.50e-14)

    CY_delta_a = fun_CY_delta_a(Ma, alpha, delta_a)
    CY_delta_e = -fun_CY_delta_a(Ma, alpha, delta_e)
    CY_delta_r = -1.43e-18 + alpha * 4.86e-20 + Ma * 1.86e-19 + delta_r * 3.84e-04 \
                 - (alpha * delta_r) * 1.17e-05 - (Ma * delta_r) * 1.07e-05 \
                 + ((alpha * Ma) * delta_r) * 2.60e-07

    CY = -CY_beta * beta + CY_delta_a + CY_delta_e + CY_delta_r

    return CY


def fun_Cl(Ma, alpha, beta, delta_a, delta_e, delta_r, p, q, r, b, c, V, gain, pull):
    Cl_beta = -1.402e-01 + Ma * 3.326e-02 - alpha * 7.590e-04 + (alpha * Ma) * 8.596e-06 \
              + (Ma ** 2) * (-3.794e-03) + (alpha ** 2) * 2.354e-06 - ((alpha * Ma) ** 2) * 1.044e-08 \
              + (Ma ** 3) * 2.219e-04 - (alpha ** 3) * 8.964e-18 - (Ma ** 4) * 6.462e-06 \
              + (alpha ** 4) * 3.803e-19 + (Ma ** 5) * 7.419e-08 - (alpha ** 5) * 3.353e-21

    def fun_Cl_delta_a(Ma, alpha, delta_a):
        return 3.570e-04 - alpha * 9.569e-05 - Ma * 3.598e-05 + delta_a * 1.170e-04 * gain \
               + ((alpha * Ma) * delta_a) * 2.794e-08 + (alpha ** 2) * 4.950e-06 \
               + (Ma ** 2) * 1.411e-06 - (delta_a ** 2) * 1.160e-06 \
               - ((alpha * Ma) * delta_a) ** 2 * 4.641e-11

    Cl_delta_a = fun_Cl_delta_a(Ma, alpha, delta_a)
    Cl_delta_e = -fun_Cl_delta_a(Ma, alpha, delta_e)
    Cl_delta_r = -5.0103e-19 + alpha * 6.2723e-20 + Ma * 2.3418e-20 + delta_r * 1.1441e-04 \
                 - (alpha * delta_r) * 2.684e-06 - (alpha * Ma) * 3.4201e-21 \
                 - (Ma * delta_r) * 3.5496e-06 + ((alpha * Ma) * delta_r) * 5.5547e-08
    Cl_r = 3.82e-01 - Ma * 1.06e-01 + alpha * 1.94e-03 - (alpha * Ma) * 8.15e-05 \
           + (Ma ** 2) * 1.45e-02 - (alpha ** 2) * 9.76e-06 + ((alpha * Ma) ** 2) * 4.49e-08 \
           - (Ma ** 3) * 1.02e-03 - (alpha ** 3) * 2.70e-07 + (Ma ** 4) * 3.56e-05 \
           + (alpha ** 4) * 3.19e-08 - (Ma ** 5) * 4.81e-07 - (alpha ** 5) * 1.06e-09
    Cl_p = -2.99e-01 + Ma * 7.47e-02 + alpha * 1.38e-03 - (alpha * Ma) * 8.78e-05 \
           - (Ma ** 2) * 9.13e-03 - (alpha ** 2) * 2.04e-04 - ((alpha * Ma) ** 2) * 1.52e-07 \
           + (Ma ** 3) * 5.73e-04 - (alpha ** 3) * 3.86e-05 - (Ma ** 4) * 1.79e-05 \
           + (alpha ** 4) * 4.21e-06 + (Ma ** 5) * 2.20e-07 - (alpha ** 5) * 1.15e-07

    wa = pull['wa']
    wd = pull['wd']
    wr = pull['wr']
    Cl = wa * Cl_beta * beta + wd * Cl_delta_a + wd * Cl_delta_e + wd * Cl_delta_r + Cl_r * (r * b / 2 / V) * flag_damping  * wr + Cl_p * (p * b / 2 / V) * flag_damping * wr

    return Cl


def fun_Cm(Ma, alpha, delta_a, delta_e, delta_r, p, q, r, b, c, V, gain, pull):
    Cm_a = -2.192e-02 + Ma * 7.739e-03 - alpha * 2.260e-03 + (alpha * Ma) * 1.808e-04 \
           - (Ma ** 2) * 8.849e-04 + (alpha ** 2) * 2.616e-04 - ((alpha * Ma) ** 2) * 2.880e-07 \
           + (Ma ** 3) * 4.617e-05 - (alpha ** 3) * 7.887e-05 - (Ma ** 4) * 1.143e-06 \
           + (alpha ** 4) * 8.288e-06 + (Ma ** 5) * 1.082e-08 - (alpha ** 5) * 2.789e-07

    def fun_Cm_delta_a(Ma, alpha, delta_a):
        return -5.67e-05 - alpha * 6.59e-05 - Ma * 1.51e-06 + delta_a * 2.89e-04 * gain \
               + (alpha * delta_a) * 4.48e-06 - (alpha * Ma) * 4.46e-06 - (Ma * delta_a) * 5.87e-06 \
               + ((alpha * Ma) * delta_a) * 9.72e-08

    Cm_delta_a = fun_Cm_delta_a(Ma, alpha, delta_a)
    Cm_delta_e = fun_Cm_delta_a(Ma, alpha, delta_e)
    Cm_delta_r = alpha * (-2.79e-05) - (alpha ** 2) * 5.89e-08 + (Ma ** 2) * 1.58e-03 \
                 + (alpha ** 3) * 6.42e-08 - (Ma ** 3) * 6.69e-04 - (alpha ** 4) * 2.10e-08 \
                 + (Ma ** 4) * 1.05e-04 + (delta_r ** 4) * 1.43e-07 + (alpha ** 5) * 3.14e-09 \
                 - (Ma ** 5) * 7.74e-06 - (delta_r ** 5) * 4.77e-22 - (alpha ** 6) * 2.18e-10 \
                 + (Ma ** 6) * 2.70e-07 - (delta_r ** 6) * 3.38e-10 + (alpha ** 7) * 5.74e-12 \
                 - (Ma ** 7) * 3.58e-9 + (delta_r ** 7) * 2.63e-24

    Cm_q = -1.36e+00 + Ma * 3.86e-01 + alpha * 7.85e-04 + (alpha * Ma) * 1.40e-04 \
           - (Ma ** 2) * 5.42e-02 + (alpha ** 2) * 2.36e-03 - ((alpha * Ma) ** 2) * 1.95e-06 \
           + (Ma ** 3) * 3.80e-03 - (alpha ** 3) * 1.48e-03 - (Ma ** 4) * 1.30e-04 \
           + (alpha ** 4) * 1.69e-04 + (Ma ** 5) * 1.71e-06 - (alpha ** 5) * 5.93e-06

    wa = pull['wa']
    wd = pull['wd']
    wr = pull['wr']
    Cm = Cm_a * wa + Cm_delta_a * wd + Cm_delta_e * wd + Cm_delta_r * wd + Cm_q * (q * c / 2 / V) * flag_damping * wr

    return Cm


def fun_Cn(Ma, alpha, beta, delta_a, delta_e, delta_r, p, q, r, b, c, V, gain, pull):
    Cn_beta = alpha * 6.998e-04 + Ma * 5.915e-02 + (alpha * Ma) * (-7.525e-05) \
              + (alpha ** 2) * 2.516e-04 + (Ma ** 2) * (-1.482e-02) + ((alpha * Ma) ** 2) * (-2.192e-07) \
              + (alpha ** 3) * (-1.0777e-04) + (Ma ** 3) * 1.269e-03 + ((alpha * Ma) ** 3) * 1.077e-08 \
              + (alpha ** 4) * 9.499e-06 + (Ma ** 4) * (-4.709e-05) + ((alpha * Ma) ** 4) * (-5.547e-011) \
              + (alpha ** 5) * (-2.595e-07) + (Ma ** 5) * 6.428e-07 + ((alpha * Ma) ** 5) * 8.586e-014

    def fun_Cn_delta_a(Ma, alpha, delta_a):
        return 2.10e-04 + alpha * 1.83e-05 - Ma * 3.56e-05 - delta_a * 1.30e-05 \
               - ((alpha * Ma) * delta_a) * 8.93e-08 - (alpha ** 2) * 6.39e-07 \
               + (Ma ** 2) * 8.16e-07 + (delta_a ** 2) * 1.97e-06 \
               + ((alpha * Ma) * delta_a) ** 2 * 1.41e-11

    Cn_delta_a = fun_Cn_delta_a(Ma, alpha, delta_a)
    Cn_delta_e = -fun_Cn_delta_a(Ma, alpha, delta_e)
    Cn_delta_r = 2.85e-18 - alpha * 3.59e-19 - Ma * 1.26e-19 - delta_r * 5.28e-04 * gain \
                 + (alpha * delta_r) * 1.39e-05 + (alpha * Ma) * 1.57e-20 \
                 + (Ma * delta_r) * 1.65e-05 - ((alpha * Ma) * delta_r) * 3.13e-07
    Cn_p = 3.68e-01 - Ma * 9.79e-02 + alpha * 7.61e-16 + (Ma ** 2) * 1.24e-02 \
           - (alpha ** 2) * 4.64e-16 - (Ma ** 3) * 8.05e-04 + (alpha ** 3) * 1.01e-16 \
           + (Ma ** 4) * 2.57e-05 - (alpha ** 4) * 9.18e-18 - (Ma ** 5) * 3.20e-07 \
           + (alpha ** 5) * 2.96e-19
    Cn_r = -2.41e+00 + Ma * 5.96e-01 - alpha * 2.74e-03 + (alpha * Ma) * 2.09e-04 \
           - (Ma ** 2) * 7.57e-02 + (alpha ** 2) * 1.15e-03 - ((alpha * Ma) ** 2) * 6.53e-08 \
           + (Ma ** 3) * 4.90e-03 - (alpha ** 3) * 3.87e-04 - (Ma ** 4) * 1.57e-04 \
           + (alpha ** 4) * 3.60e-05 + (Ma ** 5) * 1.96e-06 - (alpha ** 5) * 1.18e-06

    wa = pull['wa']
    wd = pull['wd']
    wr = pull['wr']
    Cn = Cn_beta * beta * wa + Cn_delta_a * wd + Cn_delta_e * wd + Cn_delta_r * wd + Cn_r * (r * b / 2 / V) * flag_damping * wr + Cn_p * (p * b / 2 / V) * flag_damping * wr

    return Cn


class Aerodynamics():
    def __init__(self, pull=None):
        self.pull = {}
        self.pull['wl'] = 1.
        self.pull['wm'] = 1.
        self.pull['wn'] = 1.
        self.pull['bl'] = 0.
        self.pull['bm'] = 0.
        self.pull['bn'] = 0.
        self.pull['wa'] = 0.7
        self.pull['wd'] = 1.3
        self.pull['wr'] = 0.7
        if pull is not None:
            self.pull = pull

    def get_pull(self, flag_pull):
        pull = copy.deepcopy(self.pull)
        if not flag_pull:
            for key in pull.keys():
                if 'w' in key:
                    pull[key] = 1
                if 'b' in key:
                    pull[key] = 0

        return pull

    def get_force(self, h, V, alpha, beta, delta_a, delta_e, delta_r, p, q, r, flag_pull=False):
        """Calculate aerodynamics force and moment in the hypersonic phase (Ma >= 2.5)
            Arguments:
                h : height [float]
                V : velocity [float]
                alpha : attack of angle [float, rad]
                beta : sideslip angle [float, rad]
                delta_a : deflection of the right elevon [float, rad]
                delta_e : deflection of the left elevon [float, rad]
                delta_r : deflection of the rudder [float, rad]
                p : roll rate [float, rad/s]
                q : pitch rate [float, rad/s]
                r : yaw rate [float, rad/s]
                flag_pull : flag whether to pull the model parameters [bool]
            Returns:
                L : lift force [float]
                D : drag force [float]
                Y : side force (right) [float]
                M_roll : roll moment [float]
                M_pitch : pitch moment [float]
                M_yaw : yaw moment [float]
            """

        dp = Earth.dynamic_pressure(h, V)

        # Geometry parameters
        S_ref = 3603  # reference area
        c = 80  # mean aerodynamic chord
        b = 60  # span

        CL, CD, CY, Cl, Cm, Cn = self.get_C(alpha, beta, delta_a, delta_e, delta_r, p, q, r, V, h, flag_pull=flag_pull)

        L = dp * S_ref * CL
        D = dp * S_ref * CD
        Y = dp * S_ref * CY
        M_roll = dp * S_ref * b * Cl
        M_pitch = dp * S_ref * c * Cm
        M_yaw = dp * S_ref * b * Cn
        # print(f'L = {L}, D = {D}, Y = {Y}, M_roll = {M_roll}, M_pitch = {M_pitch}, M_yaw = {M_yaw}')

        return Np2Num(L), Np2Num(D), Np2Num(Y), Np2Num(M_roll), Np2Num(M_pitch), Np2Num(M_yaw)

    def get_C(self, alpha, beta, delta_a, delta_e, delta_r, p, q, r, V, h, flag_pull=True):
        # Unit conversion: except for the sideslip angle, all other angles and angular rates are in degrees
        alpha = alpha * 57.3
        delta_a, delta_e, delta_r = delta_a * 57.3, delta_e * 57.3, delta_r * 57.3
        p, q, r = p * 57.3, q * 57.3, r * 57.3
        earth = Earth()
        rho = earth.rho(h)
        Ma = V / earth.V_sound(h)

        # Limit the quantities within the valid range
        Ma = np.clip(Ma, 2.5, 24)
        alpha = np.clip(alpha, -4., 12.)
        delta_a = np.clip(delta_a, -20., 20.)
        delta_e = np.clip(delta_e, -20., 20.)
        delta_r = np.clip(delta_r, -20., 20.)

        # Gain of control effectiveness
        gain = 1.

        # Geometry parameters
        c = 80  # mean aerodynamic chord
        b = 60  # span

        pull = self.get_pull(flag_pull)
        # aerodynamic coefficients
        CL = fun_CL(Ma, alpha, delta_a, delta_e)
        CD = fun_CD(Ma, alpha, delta_a, delta_e, delta_r)
        CY = fun_CY(Ma, alpha, beta, delta_a, delta_e, delta_r)
        Cl = fun_Cl(Ma, alpha, beta, delta_a, delta_e, delta_r, p, q, r, b, c, V, gain, pull) * pull['wl'] + pull['bl']
        Cm = fun_Cm(Ma, alpha, delta_a, delta_e, delta_r, p, q, r, b, c, V, gain, pull) * pull['wm'] + pull['bm']
        Cn = fun_Cn(Ma, alpha, beta, delta_a, delta_e, delta_r, p, q, r, b, c, V, gain, pull) * pull['wn'] + pull['bn']

        return CL, CD, CY, Cl, Cm, Cn


    def Calculate_fL_gL(self, h, V, flag_pull=False):
        """Calculate fL and gL
        Arguments:
            h : height [float]
            V : velocity [float]
            flag_pull : flag whether to pull the model parameters [bool]
        Returns:
            fL : fL [float]
            gL : gL [float]
        """
        alpha_min, alpha_max = 4. / 57.3, 12. / 57.3
        L_min, _, _, _, _, _, = self.get_force(h, V, alpha_min, 0., 0., 0., 0., 0., 0., 0., flag_pull)
        L_max, _, _, _, _, _, = self.get_force(h, V, alpha_max, 0., 0., 0., 0., 0., 0., 0., flag_pull)

        gL = (L_max - L_min) / (alpha_max - alpha_min)
        fL = L_min - gL * alpha_min

        return fL, gL

    def Calculate_fM_GM(self, h, V, alpha, beta, p, q, r, flag_pull=False):
        """Calculate fM and gM
        Arguments:
            h : height [float]
            V : velocity [float]
            alpha : angle of attack [float]
            beta : sideslip angle [float]
            p : roll rate [float]
            q : pitch rate [float]
            r : yaw rate [float]
            flag_pull : flag whether to pull the model parameters [bool]
        Returns:
            fM : fM [float]
            gM : gM [float]
        """

        def fun_moment(da, de, dr):
            L, D, Y, M_roll, M_pitch, M_yaw = self.get_force(h, V, alpha, beta, da, de, dr, p, q, r, flag_pull)
            return M_roll, M_pitch, M_yaw

        f1, f2, f3 = fun_moment(0., 0., 0.)
        fM = np.array([f1, f2, f3]).reshape(-1, 1)

        dmin, dmax = -10. / 57.3, 10. / 57.3
        M1_min, M2_min, M3_min = fun_moment(dmin, 0., 0.)
        M1_max, M2_max, M3_max = fun_moment(dmax, 0., 0.)
        g1 = np.array([M1_max, M2_max, M3_max]) - np.array([M1_min, M2_min, M3_min])

        M1_min, M2_min, M3_min = fun_moment(0., dmin, 0.)
        M1_max, M2_max, M3_max = fun_moment(0., dmax, 0.)
        g2 = np.array([M1_max, M2_max, M3_max]) - np.array([M1_min, M2_min, M3_min])

        M1_min, M2_min, M3_min = fun_moment(0., 0., dmin)
        M1_max, M2_max, M3_max = fun_moment(0., 0., dmax)
        g3 = np.array([M1_max, M2_max, M3_max]) - np.array([M1_min, M2_min, M3_min])

        gM = np.hstack((g1.reshape(-1, 1), g2.reshape(-1, 1), g3.reshape(-1, 1))) / (dmax - dmin)

        return fM, gM


def Thrust(h, V, thr):
    """Calculate thrust
    Arguments:
        h : height [float]
        V : velocity [float]
        thr : throttle angle [float]
    Returns:
        T : thrust [float]
    """

    S_ref = 3603  # reference area
    earth = Earth()
    rho = earth.rho(h)
    CT = 0.02576 * thr if thr < 1 else 0.0224 + 0.00336 * thr
    T = 0.5 * rho * V**2 * S_ref * CT

    return T




