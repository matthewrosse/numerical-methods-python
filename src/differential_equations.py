from typing import Callable, List
import math
import matplotlib.pyplot as plt
import numpy as np


def euler_method(
    f: Callable, x0: float, y0: float, h: float, samples: int
) -> List[float]:
    result: List[float] = [y0]

    for _ in range(1, samples):
        yi = y0 + h * f(x0, y0)
        result.append(yi)

        y0 = yi
        x0 += h

    return result


def heun_method(
    f: Callable, x0: float, y0: float, h: float, samples: int
) -> List[float]:
    result: List[float] = [y0]

    for _ in range(1, samples):
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + h, y0 + k1)

        yi = y0 + (k1 + k2) / 2

        result.append(yi)

        y0 = yi
        x0 += h

    return result


def runge_kutty(
    f: Callable, x0: float, y0: float, h: float, samples: int
) -> List[float]:
    result: List[float] = [y0]

    for _ in range(1, samples):
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + h / 2, y0 + k1 / 2)
        k3 = h * f(x0 + h / 2, y0 + k2 / 2)
        k4 = h * f(x0 + h, y0 + k3)

        yi = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        result.append(yi)
        y0 = yi
        x0 += h

    return result


f: Callable = lambda x, y: -y - (math.e ** (-x))
# h = 10e-3
h = 0.5
t = np.arange(0.0, 5.0, h)
samples = t.size
euler_result = euler_method(f, 0, 1, h, samples)

heun_result = heun_method(f, 0, 1, h, samples)

runge_kutty_result = runge_kutty(f, 0, 1, h, samples)

plt.plot(t, euler_result, "r-", label="Euler")
plt.plot(t, heun_result, "b-", label="Heun")

plt.plot(t, runge_kutty_result, "g-", label="Runge kutty")
plt.grid(True)
plt.show()
