import numpy as np

def Np2Num(x):
    return np.array(x).ravel()[0]

def RKM_4(y, f, h):
    # Runge-Kutta methods (4th order)
    # Function: Performs 4th order Runge-Kutta integration
    # Input: y is the initial value; f is the differential function; h is the integration step size
    # Output: Y is the integration result
    if h == 0:
        return y

    K0 = y
    K1 = f(y)
    K2 = f(y + 0.5 * h * K1)
    K3 = f(y + 0.5 * h * K2)
    K4 = f(y + h * K3)
    Y = K0 + h / 6 * (K1 + K2 * 2 + K3 * 2 + K4)

    Y = y + f(y) * h

    return Y


