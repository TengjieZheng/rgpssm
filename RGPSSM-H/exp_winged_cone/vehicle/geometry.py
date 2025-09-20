import numpy as np

def asin(x):
    x = np.array(x).ravel()[0]
    x = np.clip(x, -1., 1.)
    return np.arcsin(x)

def mat_trans(axis, angle):
    """Coordinate transformation matrix
    Arguments:
        axis : axis of rotation [int: 0, 1, 2]
        angle : rotation angle [float, rad]
    Returns:
        mat : Rotation matrix [type]
    """
    angle = np.array(angle).ravel()[0]
    sina = np.sin(angle)
    cosa = np.cos(angle)

    if axis == 0:
        mat = [[1., 0., 0.], [0., cosa, sina], [0., -sina, cosa]]
    elif axis == 1:
        mat = [[cosa, 0., -sina], [0., 1., 0.], [sina, 0., cosa]]
    elif axis == 2:
        mat = [[cosa, sina, 0.], [-sina, cosa, 0.], [0., 0., 1.]]
    mat = np.array(mat)

    return mat

def get_aerodynamic_angles(gamma, chi, roll, pitch, yaw):
    """
    Arguments:
        gamma : flight path angle [float, rad]
        chi : heading angle [float, rad]
        roll : roll angle [float, rad]
        pitch : pitch angle [float, rad]
        yaw : yaw angle [float, rad]
    Returns:
        alpha : attack of angle [float, rad]
        beta : sideslip angle [float, rad]
        phi_v : bank angle [float, rad]
    """
    Lkb = mat_trans(1, gamma) @ mat_trans(2, chi) @ mat_trans(2, -yaw) @ mat_trans(1, -pitch) @ mat_trans(0, -roll)
    beta = asin(Lkb[0, 1])
    alpha = asin(Lkb[0, 2] / np.cos(beta))
    phi_v = asin(Lkb[2, 1] / np.cos(beta))

    return alpha, beta, phi_v

def get_attidude_angles(gamma, chi, alpha, beta, phi_v):
    """
    Arguments:
        gamma : flight path angle [float, rad]
        chi : heading angle [float, rad]
        alpha : attack of angle [float, rad]
        beta : sideslip angle [float, rad]
        phi_v : bank angle [float, rad]
    Returns:
        roll : roll angle [float, rad]
        pitch : pitch angle [float, rad]
        yaw : yaw angle [float, rad]
    """
    Lbu = mat_trans(1, alpha) @ mat_trans(2, -beta) @ mat_trans(0, phi_v) @ mat_trans(1, gamma) @ mat_trans(2, chi)
    pitch = -asin(Lbu[0, 2])
    yaw = asin(Lbu[0, 1] / np.cos(pitch))
    roll = asin(Lbu[1, 2] / np.cos(pitch))

    return roll, pitch, yaw



if __name__ == '__main__':
    gamma, chi = 0., 0.
    roll, pitch, yaw = get_attidude_angles(gamma, chi, 2./57.3, 0., -40/57.3)
    alpha, beta, phi_v = get_aerodynamic_angles(gamma, chi, roll, pitch, yaw)
    print(roll * 57.3, pitch * 57.3, yaw * 57.3)
    print(alpha * 57.3, beta * 57.3, phi_v * 57.3)