from collections.abc import Callable
import random as rd
from typing import List


class Integral:
    @staticmethod
    def stochastic(func: Callable, a, b, samples):
        coeff = (b - a) / samples
        result = 0
        for _ in range(1, samples + 1):
            result += func(rd.uniform(a, b))

        return coeff * result

    @staticmethod
    def rectangle_method(func: Callable, a, b, n: int):
        h = (b - a) / n
        x = a
        sum = 0

        for _ in range(n + 1):
            sum += func(x)
            x += h

        return h * sum

    @staticmethod
    def trapezoid_method(func: Callable, a, b, n: int):
        h = (b - a) / n
        sum = 0
        x = a

        for _ in range(n):
            sum += func(x) + func(x + h)
            x += h

        return (h / 2) * sum

    @staticmethod
    def simpson_method(func: Callable, a, b, n: int):
        h = (b - a) / n
        sum = 0
        x = a

        for _ in range(n):
            sum += func(x) + 4 * func((x + x + h) / 2) + func(x + h)
            x += h

        return h / 6 * sum

    @staticmethod
    def gauss_quadrature(
        func: Callable, a, b, samples: int, scales: List[float], nodes: List[float]
    ):
        h = (b - a) / 2
        result = 0

        for i in range(samples):
            result += scales[i] * func(((b - a) / 2) * nodes[i] + (b - a) / 2)

        return result * h

    @staticmethod
    def complex_quadrature(func: Callable, a, b, n, N, scales, nodes):
        h = (b - a) / 2
        result = 0
        c = 0
        d = 0
        e = 0

        for i in range(N):
            c = a + i * h
            d = a + (i + 1) * h
            for j in range(n):
                e += scales[j] * func((h / 2) * nodes[j] + ((c + d) / 2))
            result += (h / 2) * e

        return result
