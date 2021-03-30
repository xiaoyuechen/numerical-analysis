# Discuss the dependence of initial values:
# I did not find any difference in the dependence of initial values.
# They both converges for the given function and intervals.
# However, according to https://en.wikipedia.org/wiki/Secant_method,
# Scant method might not converge if f'=0 is on [x0, x1].
# I tried with x0=4 and x1=6 for the first equation.
# Clearly f'=0 is in [x0, x1], but Secant still converges.

# Note that we use recursion instead of iterations in the implementation.
# The reason is to improve the readability of the code.

from math import exp


def newton(x0, epsilon, imax, f, df):
    if imax == 0:
        return x0
    root = x0 - f(x0) / df(x0)
    if abs(root - x0) < epsilon:
        return root
    return newton(root, epsilon, imax - 1, f, df)


def secant(x0, x1, epsilon, imax, f):
    if imax == 0:
        return x1
    root = x1 - f(x1) * (x1-x0) / (f(x1)-f(x0))
    if abs(root - x1) < epsilon:
        return root
    return secant(x1, root, epsilon, imax - 1, f)


def f1(x): return x**4 - 5*(x**3) + 9*x + 3
# Derivative of f1
def df1(x): return 4*(x**3) - 15*(x**2) + 9


def f2(x): return exp(x) - 2*(x**2) - 5
# Derivative of f2
def df2(x): return exp(x) - 4*x


# This is just a helper function to print newton results
def print_newton(fn, x0, root):
    print(f"Newton on {fn.__name__}: root={root} x0={x0}")


# This is just a helper function to print secant results
def print_secant(fn, x0, x1, root):
    print(f"Secant on {fn.__name__}: root={root} x0={x0} x1={x1}")


def main():
    # parameters we use
    epsilon = 0.000001
    imax = 10000
    f1_newton_x0s = [4, 4.5, 5, 5.5, 6]
    f1_secant_x0x1 = (4, 4.1)
    f2_newton_x0s = [3, 3.25, 3.5, 3.75, 4]
    f2_secant_x0x1 = (3.0, 3.1)

    # solve the 1st equation
    for x0 in f1_newton_x0s:
        root = newton(x0, epsilon, imax, f1, df1)
        print_newton(f1, x0, root)

    root = secant(f1_secant_x0x1[0], f1_secant_x0x1[1], epsilon, imax, f1)
    print_secant(f1, f1_secant_x0x1[0], f1_secant_x0x1[1], root)

    # solve the 2nd equation
    for x0 in f2_newton_x0s:
        root = newton(x0, epsilon, imax, f2, df2)
        print_newton(f2, x0, root)

    root = secant(f2_secant_x0x1[0], f2_secant_x0x1[1], epsilon, imax, f2)
    print_secant(f2, f2_secant_x0x1[0], f2_secant_x0x1[1], root)


if __name__ == "__main__":
    main()
