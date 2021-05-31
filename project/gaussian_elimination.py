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
        # find a row whose pivot has max abs value and set it as the pivot row
        max_row, max = None, 0
        for row in range(pivot, m):
            if abs(M[row][pivot]) > max:
                max_row = row
        M[max_row], M[pivot] = M[pivot], M[max_row]
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


def print_mat(M: List[List[float]]):
    # print the matrix on each step
    print("M=")
    for row in M:
        print(row)


if __name__ == "__main__":
    M = [[0.143, 0.357, 2.01], [-1.31, 0.911, 1.99], [11.2, -4.30, -0.605]]
    b = [-5.173, -5.458, 4.415]
    x = gaussian_elimination(M, b)
    print("x=")
    print(x)
