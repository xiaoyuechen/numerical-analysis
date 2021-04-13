from typing import List


def gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    """solve systems of equation using Gaussian elimination
    Parameters
    __________
    A : matrix A
    b : vector b

    Returns
    _______
    x : the solution
    x is [] if A is singular
    """
    # merge matrix A and vector b to matrix M
    M = [row.copy() for row in A]
    for i, row in enumerate(M):
        row.append(b[i])
    m = len(M)
    n = m + 1

    # use Gaussian elimination to reduce the matrix to row-echelon form
    for pivot in range(m - 1):
        # find a row with a non-zero pivot and set it as the pivot row
        for row in range(pivot, m):
            if M[row][pivot] != 0:
                M[row], M[pivot] = M[pivot], M[row]
                break
        # if pivot is 0, the matrix is singular
        if M[pivot][pivot] == 0:
            return []
        # eliminate the pivot element from all the rows below the pivot row
        for row in range(pivot + 1, m):
            f = M[row][pivot] / M[pivot][pivot]
            M[row][pivot] = 0
            for col in range(pivot + 1, n):
                M[row][col] -= f * M[pivot][col]

    # construct the solution using back substitution
    x = [row[n - 1] for row in M]
    for row in reversed(range(m)):
        # if pivot is 0, the matrix is singular
        if M[row][row] == 0:
            return []
        for col in range(row + 1, m):
            x[row] -= M[row][col] * x[col]
        x[row] /= M[row][row]
    return x


if __name__ == "__main__":
    M = [[0, 2, 1], [1, -2, -3], [-1, 1, 2]]
    b = [-8, 0, 3]
    x = gaussian_elimination(M, b)
    print(x)
