from collections.abc import Callable
from math import sqrt


def trapezoidal(f: Callable[[float], float], a: float, b: float,
                n: int) -> float:
    """Calculate an integral using the trapezoidal rule

    Parameters
    __________
    f : the function to be integrated
    a : min of the interval
    b : max of the interval
    n : the partition number

    Returns
    _______
    the integral
    """
    dx = (b - a) / n
    sum = 0.0
    for i in range(n):
        ai = a + dx * i
        bi = ai + dx
        sum += dx * (f(ai) + f(bi)) / 2
    return sum


def simpson(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Calculate an integral using the Simpson's rule

    Parameters
    __________
    f : the function to be integrated
    a : min of the interval
    b : max of the interval
    n : the partition number

    Returns
    _______
    the integral
    """
    dx = (b - a) / n
    sum = 0.0
    for i in range(n):
        ai = a + dx * i
        ci = ai + dx / 2
        bi = ai + dx
        # we use 2nd order polynomial for function approximation
        # between ai and bi
        sum += dx / 6 * (f(ai) + 4 * f(ci) + f(bi))
    return sum


def main():
    f = lambda x: sqrt(x)
    partitions = (10, 100, 1000)
    a = 0
    b = 2
    rules = (trapezoidal, simpson)

    for rule in rules:
        for n in partitions:
            s = rule(f, a, b, n)
            print(f"{rule.__name__} s={s} n={n}")


if __name__ == "__main__":
    main()

# results:
# trapezoidal s=1.8682025382318126 n=10
# trapezoidal s=1.8850418772248283 n=100
# trapezoidal s=1.8855996071060295 n=1000
# simpson s=1.8830508367710803 n=10
# simpson s=1.8855368985470358 n=100
# simpson s=1.8856155158810004 n=1000
