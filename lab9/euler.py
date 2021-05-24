import numpy as np
import matplotlib.pyplot as plt
from math import exp, floor


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
    n = floor((tn - t0) / h) + 1
    ys = np.zeros(n)
    ys[0] = y0
    for i in range(1, n):
        tn = t0 + h * (i - 1)
        ys[i] = ys[i - 1] + h * f(tn, ys[i - 1])
    return ys


if __name__ == "__main__":
    f = lambda x, y: y - x
    f_exact = lambda x: x + 1 - np.exp(x) / 2
    hs = [0.1, 0.05, 0.001]
    x0, y0 = 0, 0.5
    xn = 1
    yn_exact = f_exact(xn)

    print(f"exact y({xn})={yn_exact}")
    for h in hs:
        yn = euler(f, h, x0, y0, xn)[-1]
        print(f"h={h}  euler y({xn})={yn}")

    # exact y(1)=0.6408590857704775
    # h=0.1  euler y(1)=0.7031287699500001
    # h=0.05  euler y(1)=0.6733511474277902
    # h=0.001  euler y(1)=0.6415380338820522

    # plotting
    x = np.linspace(0, 1, 100)
    y_exact = f_exact(x)
    plt.plot(x, y_exact, label="exact")
    for h in hs:
        y = euler(f, h, x0, y0, xn)
        plt.plot(np.linspace(x0, xn, len(y)), y, label=f"euler h={h}")
    plt.legend()
    plt.show()
