import numpy as np
import matplotlib.pyplot as plt
from runge_kutta import runge_kutta4
from gaussian_elimination import gaussian_elimination


def shooting(f, h, epsilon, max_iter, t0, y0, tn, yn, guess1, guess2):
    z1, z2 = guess1, guess2
    fs = lambda t, y: np.array([y[1], f(y[0])])  # set up the two IVPs
    # o is the objective to minimize
    o1 = runge_kutta4(fs, h, t0, np.array([y0, z1]), tn)[0, -1] - yn
    o2 = runge_kutta4(fs, h, t0, np.array([y0, z2]), tn)[0, -1] - yn
    for _ in range(max_iter):
        z = z2 - o2 * (z2 - z1) / (o2 - o1)  # scant method
        r = runge_kutta4(fs, h, t0, np.array([y0, z]), tn)[0]
        o = r[-1] - yn
        z1, o1 = z2, o2
        z2, o2 = z, o
        if np.abs(o) < epsilon:
            return r
    return None


def finite_difference(p, q, r, h, t0, y0, tn, yn):
    n = int((tn - t0) / h)
    A = np.zeros((n - 1, n - 1))
    f = np.zeros(n - 1)
    for i in range(n - 1):
        ti = t0 + h * (i + 1)
        # construct matrix A
        if i - 1 >= 0:
            A[i, i - 1] = 1 + h / 2 * p(ti)
        A[i, i] = -2 - h**2 * q(ti)
        if i + 1 < n - 1:
            A[i, i + 1] = 1 - h / 2 * p(ti)

        # construct vector f
        f[i] = h**2 * r(ti)
        if i == 0: f[i] -= (1 + h / 2 * p(ti)) * y0
        if i == n - 2: f[i] -= (1 - h / 2 * p(ti)) * yn
    ys = gaussian_elimination(A.tolist(), f.tolist())
    ys.insert(0, y0)  # add the initial value for consistency
    ys.append(yn)  # add the last value for consistency
    return ys


if __name__ == "__main__":
    f = lambda x: 3 / 2 * x
    p = lambda t: 0
    q = lambda t: 3 / 2
    r = lambda t: 0
    hs = [0.1, 0.01]
    epsilon = 0.001
    max_iter = 10000

    for h in hs:
        xs = np.linspace(0, 1, int(1 / h) + 1)
        ys_sh = shooting(f, h, epsilon, max_iter, 0, 4, 1, 1, 0, 1)
        plt.plot(xs, ys_sh, label=f"shooting h={h}")
        ys_fd = finite_difference(p, q, r, h, 0, 4, 1, 1)
        plt.plot(xs, ys_fd, label=f"finite difference h={h}")
    plt.legend()
    plt.show()
