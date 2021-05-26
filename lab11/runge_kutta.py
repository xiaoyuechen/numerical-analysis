import numpy as np
import matplotlib.pyplot as plt


def runge_kutta4(f, h, t0, y0, tn):
    """use Runge-Kutta 4th order method to solve ODE at tn
    Parameters
    __________
    f : the right hand side of the ODE
    h : resolution
    t0 : the initial position
    y0 : the initial value
    tn : the unknown position
    Returns
    _______
    ys : the values including the intermediate values
         tn's value is at the end (returning the
         intermediates are continent for plotting)
    """
    n = int((tn - t0) / h) + 1
    ys = np.zeros((y0.shape[0], n))
    ys[:, 0] = y0
    for i in range(0, n - 1):
        ti = t0 + h * i
        yi = ys[:, i]
        k1 = h * f(ti, yi)
        k2 = h * f(ti + h / 2, yi + k1 / 2)
        k3 = h * f(ti + h / 2, yi + k2 / 2)
        k4 = h * f(ti + h, yi + k3)
        ys[:, i + 1] = yi + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
    return ys


if __name__ == "__main__":
    f = lambda t, y: np.array([y[1], 3 / 2 * y[0]])
    ys = runge_kutta4(f, 0.01, 0, np.array([4, -5]), 1)
    xs = np.linspace(0, 1, 101)
    plt.plot(xs, ys[0])
    plt.show()
