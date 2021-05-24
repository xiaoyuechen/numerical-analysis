import numpy as np
import matplotlib.pyplot as plt
from euler import euler
from interpolation import linear_interpolation


def euler_midpoint(f, h, t0, y0, tn):
    """use Euler method with midpoint to solve ODE at tn
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
    ys = np.zeros(n)
    ys[0] = y0
    for i in range(0, n - 1):
        ti = t0 + h * i
        yi = ys[i]
        k1 = h * f(ti, yi)
        k2 = h * f(ti + h / 2, yi + k1 / 2)
        ys[i + 1] = yi + k2
    return ys


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
    ys = np.zeros(n)
    ys[0] = y0
    for i in range(0, n - 1):
        ti = t0 + h * i
        yi = ys[i]
        k1 = h * f(ti, yi)
        k2 = h * f(ti + h / 2, yi + k1 / 2)
        k3 = h * f(ti + h / 2, yi + k2 / 2)
        k4 = h * f(ti + h, yi + k3)
        ys[i + 1] = yi + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
    return ys


def adams_bashforth(f, h, t0, y0, y1, tn):
    """use Adams-Bashforth 2nd order method to solve ODE at tn
    Parameters
    __________
    f : the right hand side of the ODE
    h : resolution
    t0 : the initial position
    y0 : the initial value
    y1 : the second initial value
    tn : the unknown position
    Returns
    _______
    ys : the values including the intermediate values
         tn's value is at the end (returning the
         intermediates are continent for plotting)
    """
    n = int((tn - t0) / h) + 1
    ys = np.zeros(n)
    ys[0], ys[1] = y0, y1
    for i in range(1, n - 1):
        ti_1, ti = t0 + h * (i - 1), t0 + h * i
        yi_1, yi = ys[i - 1], ys[i]
        ys[i + 1] = yi + h * (3 / 2 * f(ti, yi) - 1 / 2 * f(ti_1, yi_1))
    return ys


def adams_moulton(f, h, t0, y0, y1, tn):
    """use Adams-Moulton 3rd order method to solve ODE at tn
    Parameters
    __________
    f : the right hand side of the ODE
    h : resolution
    t0 : the initial position
    y0 : the initial value
    y1 : the second initial value
    tn : the unknown position
    Returns
    _______
    ys : the values including the intermediate values
         tn's value is at the end (returning the
         intermediates are continent for plotting)
    """
    n = int((tn - t0) / h) + 1
    ys = np.zeros(n)
    ys[0], ys[1] = y0, y1
    for i in range(1, n - 1):
        ti_1, ti, tip1 = t0 + h * (i - 1), t0 + h * i, t0 + h * (i + 1)
        yi_1, yi = ys[i - 1], ys[i]
        yip1 = runge_kutta4(f, h, ti, yi, tip1)[-1]
        ys[i + 1] = yi + h * (5 / 12 * f(tip1, yip1) + 8 / 12 * f(ti, yi) -
                              1 / 12 * f(ti_1, yi_1))
    return ys


if __name__ == "__main__":
    # problem 1
    f = lambda t, y: 1 / t**2 - y / t - y**2
    f_exact = lambda t: -1 / t

    methods = [euler, euler_midpoint, runge_kutta4]

    h = 0.05
    t0 = 1.0
    y0 = -1.0
    tn = 2.0

    x = np.linspace(t0, tn, int((tn - t0) / h) + 1)
    y_exact = f_exact(x)
    plt.plot(x, y_exact, label="exact")
    for i, method in enumerate(methods):
        y = method(f, h, t0, y0, tn)
        plt.plot(x, y, label=f"{method.__name__} h={h}")

        print(f"{i}.b")
        ts = np.array([1.052, 1.555, 1.978])
        ys_interp = linear_interpolation(x, y, ts)
        ys_exact = f_exact(ts)
        print(f"Interp: {ys_interp}")
        print(f"Exact: {ys_exact}")

    plt.legend()
    plt.show()

    # problem 2
    f = lambda x, y: y - x**2
    f_exact = lambda x: 2 + 2 * x + x**2 - np.exp(x)
    h = 0.1
    t0 = 0.0
    y0 = f_exact(t0)
    y1 = f_exact(t0 + h)
    tn = 3.3
    x = np.linspace(t0, tn, int((tn - t0) / h) + 1)
    y_exact = f_exact(x)
    plt.plot(x, y_exact, label="exact")
    y_rk4 = runge_kutta4(f, h, t0, y0, tn)
    plt.plot(x, y_rk4, label="Runge-Kutta 4")
    y_ab = adams_bashforth(f, h, t0, y0, y1, tn)
    plt.plot(x, y_ab, label="Adams-Bashforth 2")
    y_am = adams_moulton(f, h, t0, y0, y1, tn)
    plt.plot(x, y_am, label="Adams-Moulton 3")
    plt.title("Problem 2")
    plt.legend()
    plt.show()
