from collections.abc import Callable
import random as rd


class Integral:
    @staticmethod
    def stochastic(func: Callable, a, b, samples):
        coeff = (b - a) / samples
        result = 0
        for _ in range(1, samples + 1):
            result += func(rd.uniform(0, 1))

        return coeff * result

    @staticmethod
    def rectangle_method(func: Callable, a, b, n: int):
        h = (b - a) / n
        x = (a + b) / 2
        sum = 0

        for _ in range(n + 1):
            sum += func(x)
            x += h

        return h * sum

    @staticmethod
    def trapezoid_method(func: Callable, a, b, n: int):
        h = (b - a) / 2
        sum = 0
        x = (a + b) / 2

        for _ in range(n):
            sum += func(x) + func(x + h)
            x += h

        return (h / 2) * sum

    @staticmethod
    def simpson_method(func: Callable, a, b, n: int):
        h = (b - a) / n
        sum = 0
        x = (a + b) / 2

        for _ in range(n):
            sum += func(x) + 4 * func((x + x + h) / 2) + func(x + h)
            x += h

        return h / 6 * sum

    # @staticmethod
    # def gauss(n):
    #     tmp = 0
    #     for i in range(0, n):
    #         x = (a + b) / 2.0 + (b - a) / 2.0 * g[i][1]
    #         tmp += g[i][0] * f(x)
    #
    # return (b - a) / 2.0 * tmp
