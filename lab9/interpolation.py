import numpy as np
import matplotlib.pyplot as plt


def linear_interpolation(x: np.array, y: np.array, xq: np.array) -> np.array:
    """linear interpolate xq
    Parameters
    __________
    x : known x, must be sorted
    y : known y corresponding to x
    xq : the xs to interpolate on
    Returns
    _______
    yq : the interpolated ys
    """
    yq = np.zeros(xq.shape)
    for i in range(len(xq)):
        # find the index of the known x to the left and right
        idxr = np.searchsorted(x, xq[i])
        idxl = idxr - 1
        # we need to fix the index in case xq is out of the known range
        if idxr == 0:
            idxl, idxr = 0, 1
        elif idxr > len(x):
            idxl, idxr = len(x) - 2, len(x) - 1
        m = (y[idxr] - y[idxl]) / (x[idxr] - x[idxl])
        yq[i] = y[idxl] + m * (xq[i] - x[idxl])
    return yq


def lagrange_interpolation(x: np.array, y: np.array, xq: np.array) -> np.array:
    """Lagrange interpolate xq
    Parameters
    __________
    x : known x
    y : known y corresponding to x
    xq : the xs to interpolate on
    Returns
    _______
    yq : the interpolated ys
    """
    yq = np.zeros(xq.shape)
    for k in range(len(xq)):
        for i in range(len(x)):
            li = 1  # Li(x)
            for j in range(len(x)):
                if i != j:
                    li *= (xq[k] - x[j]) / (x[i] - x[j])
            yq[k] += li * y[i]
    return yq


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 8, 6, 10])
    y = np.array([2, 2.5, 7, 10.5, 12.75, 13, 13])
    xq = np.linspace(1, 10, 100)  # the sample points for plotting
    interps = [linear_interpolation, lagrange_interpolation]
    for interp in interps:
        yq = interp(x, y, xq)
        plt.plot(xq, yq)
        plt.title(interp.__name__)
        plt.grid()
        plt.show()
