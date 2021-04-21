import numpy as np


def qr_decompose(A):
    """using Gram-Schmidt to decompose a square matrix to QR
    Parameters
    __________
    A: a rank-n nxn matrix

    Returns
    _______
    Q: the orthonormal matrix
    R: the upper triangular matrix
    """
    n = A.shape[0]
    Q, R = np.zeros((n, n)), np.zeros((n, n))
    # i: the currently processing column index
    for i in range(n):
        # p: the projection on processed columns
        p = np.zeros((1, n))
        # j: the processed column index
        for j in range(i):
            R[j, i] = np.matmul(Q[:, j].T, A[:, i])
            p += R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(A[:, i] - p)
        Q[:, i] = (A[:, i] - p) / R[i, i]
    return Q, R


if __name__ == "__main__":
    A = np.array([[0, 1, 1], [1, 1, 2], [0, 0, 3]])
    Q, R = qr_decompose(A)
    print("Q")
    print(Q)
    print("R")
    print(R)
