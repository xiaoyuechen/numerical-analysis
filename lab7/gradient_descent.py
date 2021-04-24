import numpy as np


def gradient_descent(f, gradient, epsilon, M, gamma, start_x):
    x = start_x
    y = f(start_x)
    for _ in range(M):
        next_x = x - gamma * gradient(x)
        next_y = f(next_x)
        if np.linalg.norm(y - next_y) < epsilon:
            break
        x = next_x
        y = next_y
    return x, y


if __name__ == "__main__":
    f = lambda x: (1 - x[0])**2 + 5 * (x[1] - x[0]**2)**2
    gf = lambda x: np.array([
        -4 * 5 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),  #
        2 * 5 * (x[1] - x[0]**2)
    ])
    x_min, y_min = gradient_descent(f, gf, 0.000000001, 100000, 0.01,
                                    np.array([-1.4, 2]))
    print(x_min)
    print(y_min)
