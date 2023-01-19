import math
from typing import Callable, Tuple


def bisection_method(f: Callable, scope: Tuple, epsilon: float = 10e-6) -> float:
    a = scope[0]
    b = scope[1]

    if f(a) * f(b) >= 0:
        raise Exception("Wrong range! f(a) * f(b) is >= 0")

    c = a

    while b - a >= epsilon:
        c = (a + b) / 2

        if abs(f(c)) <= epsilon:
            return c

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c


def falsi_method(
    f: Callable, scope: Tuple, iterations: int = 10000000, epsilon: float = 10e-6
) -> float:
    a = scope[0]
    b = scope[1]

    if f(a) * f(b) >= 0:
        raise Exception("Wrong range! f(a) * f(b) is >= 0")

    c = a

    for _ in range(iterations):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))

        if math.fabs(f(c)) <= epsilon:
            return c

        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c

    return c


func: Callable = lambda x: 2 * x**2 + 4 * x - 1.0
scope = (-3, -2)
epsilon = 10e-4

print("BISECTION:")
print(bisection_method(func, scope, epsilon))
print("FALSI:")
print(falsi_method(func, scope, 1000000000, epsilon))
