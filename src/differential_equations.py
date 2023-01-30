from typing import Callable, List
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import interactive


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


def euler_prim_method(f: Callable, x0: float, x0_prime: float, h: float, samples: int):
    result: List[float] = [x0]
    result_prim: List[float] = [x0_prime]

    for i in range(1, samples):
        result_prim.append(
            result_prim[i - 1] + h * f(result[i - 1], result_prim[i - 1])
        )
        result.append(result[i - 1] + h * result_prim[i])

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

plt.figure(1)

plt.plot(t, euler_result, "r-", label="Euler")
plt.plot(t, heun_result, "b-", label="Heun")

plt.plot(t, runge_kutty_result, "g-", label="Runge kutty")
plt.legend(["Euler", "RK2 (Heun)", "RK4 Runge-Kutty"])
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

interactive(True)

plt.show()

plt.figure(2)

func_cw3: Callable = lambda x, _: -2 * x**3 + 12 * x**2 - 20 * x + 8.5
t = np.arange(0.0, 4.0, h)

euler_result = euler_method(func_cw3, 0, 1, h, t.size)
heun_result = heun_method(func_cw3, 0, 1, h, t.size)
runge_kutty_result = runge_kutty(func_cw3, 0, 1, h, t.size)

plt.plot(t, euler_result, "r-", label="Euler")
plt.plot(t, heun_result, "b-", label="Heun")

plt.plot(t, runge_kutty_result, "g-", label="Runge kutty")
plt.legend(["Euler", "RK2 (Heun)", "RK4 Runge-Kutty"])
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

plt.show()

plt.figure(3)

func_cw4: Callable = lambda _, t: -0.0245 * (t - 300)
# x0 = 0, y0 = 550, samples = 60, h = 1

h = 1

t = np.arange(0.0, 60.0, h)

euler_result = euler_method(func_cw4, 0, 550, h, t.size)
heun_result = heun_method(func_cw4, 0, 550, h, t.size)
runge_kutty_result = runge_kutty(func_cw4, 0, 550, h, t.size)

plt.plot(t, euler_result, "r-", label="Euler")

plt.plot(t, heun_result, "b-", label="Heun")

plt.plot(t, runge_kutty_result, "g-", label="Runge kutty")

plt.legend(["Euler", "RK2 (Heun)", "RK4 Runge-Kutty"])
plt.xlabel("Czas [s]")
plt.ylabel("Temperatura (stopnie Celsjusza)")
plt.grid(True)


plt.figure(4)

func_cw5: Callable = lambda m, t: 2.77 * np.log(541 / (541 - (m * 1.26)))

h = 1

t = np.arange(0.0, 40.0, h)

euler_result = euler_method(func_cw5, 0, 0, h, t.size)
heun_result = heun_method(func_cw5, 0, 0, h, t.size)
runge_kutty_result = runge_kutty(func_cw5, 0, 0, h, t.size)

plt.plot(t, euler_result, "r-", label="Euler")

plt.plot(t, heun_result, "b-", label="Heun")

plt.plot(t, runge_kutty_result, "g-", label="Runge kutty")

plt.legend(["Euler", "RK2 (Heun)", "RK4 Runge-Kutty"])
plt.xlabel("Czas [s]")
plt.ylabel("Wysokość [km]")
plt.grid(True)

# plt.show()
#
# plt.figure(5)
#
# func_cw8: Callable = lambda x, x_prime: -2.0 * 1.8 * x_prime + 10**2 * x
#
# h = 1
#
# t = np.arange(0.0, 6.0, h)
#
# euler_result = euler_prim_method(func_cw8, 1, 0, h, t.size)
# # heun_result = heun_method(func_cw8, 0, 0, h, t.size)
# # runge_kutty_result = runge_kutty(func_cw8, 0, 0, h, t.size)
#
# plt.plot(t, euler_result, "r-", label="Euler")
#
# # plt.plot(t, heun_result, "b-", label="Heun")
# #
# # plt.plot(t, runge_kutty_result, "g-", label="Runge kutty")
#
# # plt.legend(["Euler", "RK2 (Heun)", "RK4 Runge-Kutty"])
# # plt.xlabel("Czas [s]")
# # plt.ylabel("Wysokość [km]")
# plt.grid(True)


interactive(False)
plt.show()
