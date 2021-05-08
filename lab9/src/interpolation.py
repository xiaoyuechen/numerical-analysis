import numpy as np
import matplotlib.pyplot as plt


def linear_interpolation(x: np.array, y: np.array, xq: np.array) -> np.array:
    yq = np.zeros(xq.shape)
    for i, xqi in enumerate(xq):
        idxr = np.searchsorted(x, xqi)
        idxl = idxr - 1
        if idxr == 0:
            idxl, idxr = 0, 1
        elif idxr > x.shape[0] - 1:
            idxl, idxr = x.shape[0] - 2, x.shape[0] - 1
        m = (y[idxr] - y[idxl]) / (x[idxr] - x[idxl])
        yq[i] = y[idxl] + m * (xqi - x[idxl])
    return yq


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 8, 6, 10])
    y = np.array([2, 2.5, 7, 10.5, 12.75, 13, 13])
    xq = np.linspace(1, 10, 100)
    yq = linear_interpolation(x, y, xq)
    plt.plot(xq, yq)
    plt.title("Linear interpolation")
    plt.grid()
    plt.show()
