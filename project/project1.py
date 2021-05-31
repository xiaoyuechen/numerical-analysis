import numpy as np
import matplotlib.pyplot as plt


def euler(f, h, t0, y0, tn):
    """use Euler method to solve ODE at tn
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
    for i in range(1, n):
        tn = t0 + h * (i - 1)
        ys[i] = ys[i - 1] + h * f(tn, ys[i - 1])
    return ys


def runge_kutta2(f, h, t0, y0, tn):
    """use Runge-Kutta 2nd order method to solve ODE at tn
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


if __name__ == "__main__":
    f = lambda t, u: np.cos(np.pi * t) + u
    u_exact = lambda t: (2 + (1 + np.pi * np.exp(-t) * np.sin(np.pi * t) - np.
                              exp(-t) * np.cos(np.pi * t)) /
                         (1 + np.pi**2)) * np.exp(t)
    methods = [euler, runge_kutta2, runge_kutta4, adams_bashforth]
    Ns = [10, 20, 40, 80, 160, 320, 640]
    t0, tn = 0, 2
    u0 = 2

    # plot solutions using different methods
    h = 0.1
    ts = np.linspace(0, 2, int(2 / h) + 1)
    us_exact = u_exact(ts)
    plt.plot(ts, us_exact, label="exact")
    for method in methods:
        us = None
        if method is adams_bashforth:
            u1 = u_exact(t0 + h)
            us = method(f, h, t0, u0, u1, tn)
        else:
            us = method(f, h, t0, u0, tn)
        plt.plot(ts, us, label=method.__name__)
    plt.legend()
    plt.title("Solutions using different methods (h=0.1)")
    plt.show()

    # plot errors using different methods
    bar_width = 0.2
    idx = np.arange(len(Ns))
    for i, method in enumerate(methods):
        errors = []
        for N in Ns:
            h = 2 / N
            us = None
            if method is adams_bashforth:
                u1 = u_exact(t0 + h)
                us = method(f, h, t0, u0, u1, tn)
            else:
                us = method(f, h, t0, u0, tn)
            error = -np.log10(np.abs(us[-1] - us_exact[-1]))
            errors.append(error)
        bars = plt.bar(idx + bar_width * i,
                       errors,
                       bar_width,
                       label=method.__name__)
        for bar in bars:
            error = bar.get_height()
            plt.text(bar.get_x(), error + .01, f"{error:.2f}")
    plt.xlabel('N')
    plt.ylabel('-lg(error)')
    plt.xticks(idx + bar_width * len(methods) / 2, Ns)
    plt.legend()
    plt.title("Errors of different methods")
    plt.show()
