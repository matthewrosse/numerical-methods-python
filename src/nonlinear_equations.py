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


def secant_method(
    f: Callable, scope: Tuple, iterations: int = 1000000, epsilon: float = 10e-6
):

    x0 = scope[0]
    x1 = scope[1]
    i = 1
    while abs(x0 - x1) >= epsilon:
        x2 = x1 - (f(x1) / (f(x1) - f(x0)) * (x1 - x0))
        i += 1
        x0 = x1
        x1 = x2

    return x1


def newton_method(f1: Callable, f2: Callable, scope: Tuple, epsilon: float = 10e-6):
    x0 = (scope[0] + scope[1]) / 2
    i = 0

    while True:
        x1 = x0 - f1(x0) / f2(x0)
        d = abs(x0 - x1)
        i += 1
        x0 = x1
        if d < epsilon:
            break

    return x0


func1: Callable = lambda x: 2 * x**2 + 4 * x - 1.0
func1_derivative: Callable = lambda x: 4 * x + 4
scope = (-3, -2)
epsilon = 10e-4

print("Funkcja 1:")
print()
print("BISECTION:")
print(bisection_method(func1, scope, epsilon))
print("FALSI:")
print(falsi_method(func1, scope, 1000000000, epsilon))
print("SECANT:")
print(secant_method(func1, scope, 1000000000, epsilon))

print("NEWTON:")
print(newton_method(func1, func1_derivative, scope, epsilon))

func2: Callable = lambda x: x + math.e ** math.tan(x)
func2_derivative: Callable = lambda x: 1 + math.e ** (math.tan(x)) / (
    math.cos(x) * math.cos(x)
)

scope2 = (-1.5, 0)
epsilon2 = 10e-5

print("Funkcja 2:")
print()
print("BISECTION:")
print(bisection_method(func2, scope2, epsilon2))
print("FALSI:")
print(falsi_method(func2, scope2, 1000000000, epsilon2))
print("SECANT:")
print(secant_method(func2, scope2, 1000000000, epsilon2))

print("NEWTON:")
print(newton_method(func2, func2_derivative, scope2, epsilon2))

print()
print("Funkcja 3")
print()

func3: Callable = lambda x: 1 / x + 2 * x + 3
func3_derivative: Callable = lambda x: -1 / (x**2) + 2

scope3 = (-1.2, -0.4)
epsilon3 = 10e-5

# print("BISECTION:")
# print(bisection_method(func3, scope3, epsilon3))
# print("FALSI:")
# print(falsi_method(func3, scope3, 1000000000, epsilon3))
print("SECANT:")
print(secant_method(func3, scope3, 1000000000, epsilon3))

print("NEWTON:")
print(newton_method(func3, func3_derivative, scope3, epsilon3))

print()
print("Funkcja 4")
print()

func4: Callable = lambda x: x**3
func4_derivative: Callable = lambda x: 3 * x**2

scope4 = (-1.0, 0.5)
epsilon4 = 10e-4

print("SECANT:")
print(secant_method(func4, scope4, 1000000000, epsilon4))

print("NEWTON:")
print(newton_method(func4, func4_derivative, scope4, epsilon4))
