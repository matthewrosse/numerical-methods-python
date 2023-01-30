from typing import Callable, List
from complex import Complex
from factorial import factorial
from euler import euler
from matrix import Matrix
from integral import Integral
from linear_equations import LinearEquations
from lu import LU
import math
from interpolation import Interpolation
from derivative import Derivative
import time


class TestsRunner:
    def __init__(self) -> None:
        self.tests = [
            self.complex_numbers_test,
            self.factorial_test,
            self.euler_test,
            self.matrix_algebra_test,
            self.systems_of_linear_equations_tests,
            self.lu_decomposition_tests,
            self.interpolation_tests,
            self.derivatives_tests,
            self.integrals_tests,
        ]

    def run_all(self) -> None:
        for test in self.tests:
            test()

    def complex_numbers_test(self) -> None:
        z1 = Complex(1.23456789123456789, 4 / 3)
        z2 = Complex(1, 2)

        print("Complex numbers tests:")
        print(f"+ : z = {z1 + z2}")
        print(f"* : z = {z1 * z2}")
        print()

    def factorial_test(self) -> None:
        print("Factorial tests")
        for n in range(0, 21, 5):
            print(f"Factorial for n = {n}: {factorial(n)}")

        print()

    def euler_test(self) -> None:
        print("Euler number tests")
        for n in range(2, 20, 2):
            print(f"For {n} elements e equals: {euler(n)}")

        print()

    def matrix_algebra_test(self) -> None:
        self.matrix_multiplication_test()
        self.matrix_det_test()
        self.matrix_inverse_test()

    def matrix_multiplication_test(self) -> None:
        print("Matrix algebra tests:")
        print()

        exercise1_a = [[-1, 4, 2, -2], [1, 2, -3, 0], [-1, 0, 0, 5]]
        exercise1_b = [[2, -1], [1, 3], [-2, 0], [0, -4]]

        print("Multiplication test:")
        print()
        print("Matrix A:")
        print()
        Matrix.print(exercise1_a)
        print()
        print("Matrix B:")
        print()
        Matrix.print(exercise1_b)

        print()
        print("Result:")
        print()
        Matrix.print(Matrix.multiply(exercise1_a, exercise1_b))
        print()

    def matrix_det_test(self) -> None:
        print("Determinant test:")
        print()

        a = [[1, 3, 2], [4, -1, 2], [1, -1, 0]]

        print("Matrix A:")
        print()
        Matrix.print(a)
        print()

        print(f"Determinant result: {Matrix.sarrus_det(a)}")
        print()

        b = [
            [2, 7, -1, 3, 2],
            [0, 0, 1, 0, 1],
            [-2, 0, 7, 0, 2],
            [-3, -2, 4, 5, 3],
            [1, 0, 0, 0, 1],
        ]

        print("Matrix B:")
        print()
        Matrix.print(b)
        print()

        print(f"Determinant result: {Matrix.laplace_det(b)}")
        print()

    def matrix_inverse_test(self) -> None:
        print("Matrix inverse test:")
        print()

        a = [[1, 3, 2], [4, -1, 2], [1, -1, 0]]

        print("Matrix A:")
        print()
        Matrix.print(a)
        print()

        print("Inverse matrix result:")
        print()
        Matrix.print(Matrix.inverse(a))

    def systems_of_linear_equations_tests(self) -> None:
        print("Systems of linear equations tests:")
        print()
        self.linear_equations_inverse_matrix_method_test()
        self.linear_equations_cramer_method_tests()
        self.linear_equations_gauss_elimination_tests()
        self.linear_equations_gauss_jordan_tests()
        self.linear_equations_gauss_pivoting()
        self.linear_equations_inserving_matrix_gauss()

    def linear_equations_inverse_matrix_method_test(self) -> None:
        exercise1_matrix = [[5, -2, 3], [-2, 3, 1], [-1, 2, 3]]
        exercise1_constant_terms = [[21], [-4], [5]]
        print("Inverse matrix method test:")
        print()
        print("Matrix: ")
        print()
        Matrix.print(exercise1_matrix)
        print()
        print("Constant terms:")
        print()
        Matrix.print(exercise1_constant_terms)
        print()
        print("Result:")
        print()
        Matrix.print(
            LinearEquations.inverse_matrix_method(
                exercise1_matrix, exercise1_constant_terms
            )
        )
        print()

    def linear_equations_cramer_method_tests(self) -> None:
        print("Cramer method test:")
        print()
        print("Matrix and constant terms same as in inverse matrix method")
        print()
        exercise1_matrix = [[5, -2, 3], [-2, 3, 1], [-1, 2, 3]]
        exercise1_constant_terms = [[21], [-4], [5]]
        print("Result: ")
        print()
        Matrix.print(
            LinearEquations.cramer_method(exercise1_matrix, exercise1_constant_terms)
        )
        print()

    def linear_equations_gauss_elimination_tests(self) -> None:
        print("Gauss elimination test:")
        print()
        a = [
            [-1, 2, -3, 3, 5],
            [8, 0, 7, 4, -1],
            [-3, 4, -3, 2, -2],
            [8, -3, -2, 1, 2],
            [-2, -1, -6, 9, 0],
        ]
        b = [[56], [62], [-10], [14], [28]]

        print("Matrix: ")
        print()
        Matrix.print(a)
        print()
        print("Constant terms:")
        print()
        Matrix.print(b)

        print()
        print("Result:")
        print()
        Matrix.print(LinearEquations.gauss_elimination(a, b))
        print()

    def linear_equations_gauss_jordan_tests(self) -> None:
        print("Gauss jordan test:")
        print()
        a = [
            [-1, 2, -3, 3, 5],
            [8, 0, 7, 4, -1],
            [-3, 4, -3, 2, -2],
            [8, -3, -2, 1, 2],
            [-2, -1, -6, 9, 0],
        ]
        b = [[56], [62], [-10], [14], [28]]

        print("Matrix: ")
        print()
        Matrix.print(a)
        print()
        print("Constant terms:")
        print()
        Matrix.print(b)

        print()
        print("Result:")
        print()
        Matrix.print(LinearEquations.gauss_jordan(a, b))
        print()

    def linear_equations_gauss_pivoting(self) -> None:
        print("Gauss with pivoting test:")
        print()
        a = [[0, 1, 1], [1, 1, 1], [2, 0, -1]]
        b = [[1], [2], [0]]

        print("Matrix: ")
        print()
        Matrix.print(a)
        print()
        print("Constant terms:")
        print()
        Matrix.print(b)

        print()
        print("Result:")
        print()
        Matrix.print(LinearEquations.gauss_elimination_pivoting(a, b))
        print()

    def linear_equations_inserving_matrix_gauss(self) -> None:
        print("Inversing matrix with gauss elimination")
        print()
        a = [
            [-1, 2, -3, 3, 5],
            [8, 0, 7, 4, -1],
            [-3, 4, -3, 2, -2],
            [8, -3, -2, 1, 2],
            [-2, -1, -6, 9, 0],
        ]

        print("Matrix: ")
        print()
        Matrix.print(a)

        print()
        print("Result:")
        print()
        Matrix.print(LinearEquations.matrix_inverse_gauss_method(a))
        print()

    def lu_decomposition_tests(self) -> None:
        print("LU tests:")
        self.lu_decompose()
        self.lu_solve()
        self.lu_inversing_matrix()
        self.lu_determinant()

    def lu_decompose(self) -> None:
        print("Decomposition:")
        print()
        a = [
            [-1, 2, -3, 3, 5],
            [8, 0, 7, 4, -1],
            [-3, 4, -3, 2, -2],
            [8, -3, -2, 1, 2],
            [-2, -1, -6, 9, 0],
        ]

        print("Matrix: ")
        print()
        Matrix.print(a)
        print()
        L, U = LU.get_LU(a)
        print()
        print("L:")
        print()
        Matrix.print(L)
        print()
        print("U:")
        print()
        Matrix.print(U)
        print()

    def lu_solve(self) -> None:
        print("Solving equation:")
        print()
        a = [
            [-1, 2, -3, 3, 5],
            [8, 0, 7, 4, -1],
            [-3, 4, -3, 2, -2],
            [8, -3, -2, 1, 2],
            [-2, -1, -6, 9, 0],
        ]
        b = [[56], [62], [-10], [14], [28]]

        print("Matrix: ")
        print()
        Matrix.print(a)
        print()
        print("Constant terms:")
        print()
        Matrix.print(b)

        print()
        print("Result:")
        print()
        Matrix.print(LU.solve(a, b))
        print()

    def lu_inversing_matrix(self) -> None:
        print("Inversing matrix:")
        print()
        a = [
            [-1, 2, -3, 3, 5],
            [8, 0, 7, 4, -1],
            [-3, 4, -3, 2, -2],
            [8, -3, -2, 1, 2],
            [-2, -1, -6, 9, 0],
        ]

        print("Matrix: ")
        print()
        Matrix.print(a)
        print()

        print("Result:")
        print()
        Matrix.print(LU.inverse_matrix(a))
        print()

    def lu_determinant(self) -> None:
        a = [
            [-1, 2, -3, 3, 5],
            [8, 0, 7, 4, -1],
            [-3, 4, -3, 2, -2],
            [8, -3, -2, 1, 2],
            [-2, -1, -6, 9, 0],
        ]

        print("Matrix: ")
        print()
        Matrix.print(a)
        print()

        print(f"Determinant: {LU.det(a)}")
        print()

    def interpolation_tests(self) -> None:
        nodes = [0, 1.5, 3, 4]
        functions_a = [lambda x: 1, lambda x: x, math.cos, math.sin]
        values = [[2], [3], [1], [3]]

        interpolation_result = Interpolation.solve(nodes, functions_a, values)

        print()
        print("Interpolation 1 result: ")
        print()
        Matrix.print(interpolation_result, 8)

        functions_b = [lambda x: 1, lambda x: x, lambda x: x**2, lambda x: x**3]

        interpolation_result = Interpolation.solve(nodes, functions_b, values)

        print()
        print("Interpolation 2 result: ")
        print()
        Matrix.print(interpolation_result, 8)

    def derivatives_tests(self) -> None:
        print("Derivatives tests:")
        print()
        self.first_derivative_test_A()
        self.first_derivative_test_B()
        self.first_derivative_test_C()
        self.second_derivative_test_A()
        self.second_derivative_test_A2()
        self.second_derivative_test_B()
        self.second_derivative_test_B2()

    def first_derivative_test_A(self) -> None:
        print("1. First derivative two point ordinary")
        print()
        function = lambda x: x * math.sin(x**2) + 1
        for h in [1e-1, 1e-3, 1e-6]:
            print(
                f"Result for x = 0, h = {h}: {Derivative.two_point_ordinary(function, 0, h)}"
            )
            print(
                f"Result for x = 1, h = {h}: {Derivative.two_point_ordinary(function, 1, h)}"
            )
            print()

    def first_derivative_test_B(self) -> None:
        print("2. First derivative two point central")
        print()
        function = lambda x: x * math.sin(x**2) + 1
        for h in [1e-1, 1e-3, 1e-6]:
            print(
                f"Result for x = 0, h = {h}: {Derivative.two_point_central(function, 0, h)}"
            )
            print(
                f"Result for x = 1, h = {h}: {Derivative.two_point_central(function, 1, h)}"
            )
            print()

    def first_derivative_test_C(self) -> None:
        print("3. First derivative three point central")
        print()
        function = lambda x: x * math.sin(x**2) + 1
        for h in [1e-1, 1e-3, 1e-6]:
            print(
                f"Result for x = 0, h = {h}: {Derivative.three_point_central(function, 0, h)}"
            )
            print(
                f"Result for x = 1, h = {h}: {Derivative.three_point_central(function, 1, h)}"
            )
            print()

    def second_derivative_test_A(self) -> None:
        print("4. Second derivative three point ordinary")
        print()
        function = lambda x: x * math.sin(x**2) + 1
        for h in [1e-1, 1e-3, 1e-6]:
            print(
                f"Result for x = 0.75, h = {h}: {Derivative.second_three_point_ordinary(function, 0.75, h)}"
            )
            print()

    def second_derivative_test_A2(self) -> None:
        print("4.2. Second derivative three point central")
        print()
        function = lambda x: x * math.sin(x**2) + 1
        for h in [1e-1, 1e-3, 1e-6]:
            print(
                f"Result for x = 0.75, h = {h}: {Derivative.second_three_point_central(function, 0.75, h)}"
            )
            print()

    def second_derivative_test_B(self) -> None:
        print("5. Second derivative three point ordinary")
        print()
        function = lambda x: math.e**x
        for h in [1e-1, 1e-3, 1e-6]:
            print(
                f"Result for x = 0, h = {h}: {Derivative.second_three_point_ordinary(function, 0, h)}"
            )
            print()
            print(
                f"Result for x = 1, h = {h}: {Derivative.second_three_point_ordinary(function, 1, h)}"
            )

    def second_derivative_test_B2(self) -> None:
        print("5.2. Second derivative three point central")
        print()
        function = lambda x: math.e**x
        for h in [1e-1, 1e-3, 1e-6]:
            print(
                f"Result for x = 0, h = {h}: {Derivative.second_three_point_central(function, 0, h)}"
            )
            print()
            print(
                f"Result for x = 1, h = {h}: {Derivative.second_three_point_central(function, 1, h)}"
            )

    def stochastic_integral_test(self):
        print("Calkowanie stochastyczne: ")
        func1 = lambda x: (4 * x**3) + (5 * x**2) + 1
        func2 = lambda x: math.cos(x) + math.e**x + math.tan(x)

        samplesArr = [1, 10, 100, 1000, 10000, 100000, 1000000]
        a1 = -1
        b1 = 1
        a2 = 0
        b2 = 1

        for sample in samplesArr:
            start = time.time()
            result = Integral.stochastic(func1, a1, b1, sample)
            end = time.time()
            print(
                f"Function 1, n: {sample}, time elapsed: {end - start} seconds = {result}"
            )

        print()

        for sample in samplesArr:
            start = time.time()
            result = Integral.stochastic(func2, a2, b2, sample)
            end = time.time()
            print(
                f"Function 2, n: {sample}, time elapsed: {end - start} seconds, result = {result}"
            )

    def integrals_tests(self) -> None:
        self.stochastic_integral_test()
        self.simpson_method_test()
        self.trapezoid_method_test()
        self.cw2()
        self.cw3()
        self.cw4()
        self.cw5()
        self.cw7()

    def simpson_method_test(self) -> None:
        print()
        func: Callable = lambda x: 4 * x**3 + 5 * x**2 + 1

        for n in [1, 10, 100, 1000, 10000, 100000, 1000000]:
            start = time.time()
            result = Integral.simpson_method(func, -1, 1, n)
            end = time.time()
            print(
                f"Simpson (n = {n}), elapsed time: {end - start} second, result = {result}"
            )
        print()

    def trapezoid_method_test(self) -> None:
        print()
        func: Callable = lambda x: 4 * x**3 + 5 * x**2 + 1
        for n in [1, 10, 100, 1000, 10000, 100000, 1000000]:
            start = time.time()
            result = Integral.trapezoid_method(func, -1, 1, n)
            end = time.time()
            print(
                f"Trapezoid (n = {n}), elapsed time: {end - start} seconds, result = {result}"
            )
        print()

    def cw2(self) -> None:
        print()
        print("Cwiczenie 2")
        print()

        func: Callable = lambda x: (2 / math.sqrt(math.pi)) * math.pow(
            math.e, -(x**2)
        )
        a = 0
        b = 1

        functions: List[Callable] = [
            Integral.rectangle_method,
            Integral.trapezoid_method,
            Integral.simpson_method,
            Integral.stochastic,
        ]

        exact_result: float = 0.84270079294971486934122063508

        for samples_count in [100, 1000, 10000, 100000]:
            for function in functions:
                start = time.time()
                result = function(func, a, b, samples_count)
                end = time.time()

                print(
                    f"Function: {function.__name__}, N={samples_count} took {end - start} seconds. Result = {result}, ERR={exact_result - result}"
                )

    def cw3(self) -> None:
        print()
        print("Cwiczenie 3")
        print()

        func: Callable = lambda x: math.cos(x) + math.pow(math.e, x) + math.tan(x)
        a = 0
        b = 1

        functions: List[Callable] = [
            Integral.rectangle_method,
            Integral.trapezoid_method,
            Integral.simpson_method,
            Integral.stochastic,
        ]

        exact_result: float = 3.17537928365295618605

        for samples_count in [100, 1000, 10000, 100000]:
            for function in functions:
                start = time.time()
                result = function(func, a, b, samples_count)
                end = time.time()

                print(
                    f"Function: {function.__name__}, N={samples_count} took {end - start} seconds. Result = {result}, ERR={exact_result - result}"
                )

    def cw4(self) -> None:
        scales_n2: List[float] = [1.0, 1.0]
        nodes_n2: List[float] = [-0.577350269, 0.577350269]
        scales_n3: List[float] = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        nodes_n3: List[float] = [-0.774596669, 0.0, 0.774596669]
        scales_n4: List[float] = [0.347854845, 0.652145155, 0.652145155, 0.347854845]
        nodes_n4: List[float] = [-0.861136312, -0.339981044, 0.339981044, 0.861135312]

        print()
        print("Cw 4")
        print()

        func_cw1: Callable = lambda x: 4 * x**3 + 5 * x**2 + 1
        a_arr = [-1, 0]
        b_arr = [1, 1]
        func_cw3: Callable = lambda x: math.cos(x) + math.e**x + math.tan(x)

        functions: List[Callable] = [func_cw1, func_cw3]

        print("N=2")

        for f in range(2):
            start = time.time()
            result = Integral.gauss_quadrature(
                functions[f], a_arr[f], b_arr[f], 2, scales_n2, nodes_n2
            )
            end = time.time()

            print(f"Function {f} took {end-start}s, result = {result}")
            print(f"a={a_arr[f]}, b={b_arr[f]}")

        print("N=3")

        for f in range(2):
            start = time.time()
            result = Integral.gauss_quadrature(
                functions[f], a_arr[f], b_arr[f], 3, scales_n3, nodes_n3
            )
            end = time.time()

            print(f"Function {f} took {end-start}s, result = {result}")

        print("N=4")

        for f in range(2):
            start = time.time()
            result = Integral.gauss_quadrature(
                functions[f], a_arr[f], b_arr[f], 4, scales_n4, nodes_n4
            )
            end = time.time()

            print(f"Function {f} took {end-start}s, result = {result}")

    def cw5(self) -> None:
        print()
        print("Complex quadrature")
        print()

        scales_n2: List[float] = [1.0, 1.0]
        nodes_n2: List[float] = [-0.577350269, 0.577350269]

        func_cw1: Callable = lambda x: 4 * x**3 + 5 * x**2 + 1
        a_arr = [-1, 0]
        b_arr = [1, 1]
        func_cw3: Callable = lambda x: math.cos(x) + math.e**x + math.tan(x)

        functions: List[Callable] = [func_cw1, func_cw3]

        for i in range(2):
            start = time.time()
            result = Integral.complex_quadrature(
                functions[i], a_arr[i], b_arr[i], 2, 5, scales_n2, nodes_n2
            )
            end = time.time()

            print(f"Function {i} took {end-start}s, result = {result}")

    def cw7(self) -> None:
        print()
        print("CW7")
        print()
        correct = 149.2742360197346071383
        func: Callable = lambda t: 5 * math.sin(t) * math.sin(t)

        rectangle_samples = 5000
        rectangle_result = Integral.rectangle_method(func, 0, 60, rectangle_samples)
        rectangle_err = abs(rectangle_result - correct)

        trapezoid_samples = 1000
        trapezoid_result = Integral.trapezoid_method(func, 0, 60, trapezoid_samples)
        trapezoid_err = abs(trapezoid_result - correct)

        simpson_samples = 200
        simpson_result = Integral.simpson_method(func, 0, 60, simpson_samples)
        simpson_err = abs(simpson_result - correct)

        print(
            f"Metoda prostokatow, {rectangle_samples} probek, wynik={rectangle_result}, err={rectangle_err}"
        )
        print(
            f"Metoda trapezow, {trapezoid_samples} probek, wynik={trapezoid_result}, err={trapezoid_err}"
        )
        print(
            f"Metoda simpsona, {simpson_samples} probek, wynik={simpson_result}, err={simpson_err}"
        )
