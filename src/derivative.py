from collections.abc import Callable


class Derivative:
    @staticmethod
    def two_point_ordinary(func: Callable, x: float, h: float = 1e-6):
        return (func(x + h) - func(x)) / h

    @staticmethod
    def two_point_central(func: Callable, x: float, h: float = 1e-6):
        return (func(x + h) - func(x - h)) / 2 * h

    @staticmethod
    def three_point_central(func: Callable, x: float, h: float = 1e-6):
        return (4 * func(x + h) - 3 * func(x) - func(x + 2 * h)) / (2 * h)

    @staticmethod
    def second_three_point_ordinary(func: Callable, x: float, h: float = 1e-6):
        return (func(x) - 2 * func(x + h) + func(x + 2 * h)) / h**2

    @staticmethod
    def second_three_point_central(func: Callable, x: float, h: float = 1e-6):
        return (func(x + h) - 2 * func(x) + func(x - h)) / (h**2)
