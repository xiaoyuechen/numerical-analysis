import numpy as np


def solve_jacobi(A: np.array, b: np.array) -> np.array:
    """Solve system of linear equations using the Jacobi method
    Parameters
    __________
    A: nxn numpy array
    B: nx1 numpy array
    
    Returns
    _______
    x: nx1 numpy array
    """
    n = A.shape[0]
    x = np.random.rand(n, 1)
    convergence = 0.00001
    while True:
        next_x = np.zeros((n, 1))
        for i in range(n):
            s = 0
            for j in range(n):
                if i != j:
                    s += A[i, j] * x[j]
            next_x[i] = (b[i] - s) / A[i, i]
        if np.linalg.norm(x - next_x) < convergence:
            break
        x = next_x
    return x


if __name__ == "__main__":
    A = np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]])
    b = np.array([-1, 2, 3]).T
    x = solve_jacobi(A, b)
    print(x)
