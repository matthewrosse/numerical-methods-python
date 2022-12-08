from complex import Complex
from factorial import factorial
from euler import euler
from matrix import Matrix
from linear_equations import LinearEquations
from lu import LU
import math
from interpolation import Interpolation
from derivative import Derivative


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
        for h in [1e-3, 1e-6, 1e-9, 1e-12]:
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
        for h in [1e-3, 1e-6, 1e-9, 1e-12]:
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
        for h in [1e-3, 1e-6, 1e-9, 1e-12]:
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
        for h in [1e-1, 1e-3, 1e-6, 1e-9, 1e-12]:
            print(
                f"Result for x = 0.75, h = {h}: {Derivative.second_three_point_ordinary(function, 0.75, h)}"
            )
            print()

    def second_derivative_test_A2(self) -> None:
        print("4.2. Second derivative three point central")
        print()
        function = lambda x: x * math.sin(x**2) + 1
        for h in [1e-1, 1e-3, 1e-6, 1e-9, 1e-12]:
            print(
                f"Result for x = 0.75, h = {h}: {Derivative.second_three_point_central(function, 0.75, h)}"
            )
            print()

    def second_derivative_test_B(self) -> None:
        print("5. Second derivative three point ordinary")
        print()
        function = lambda x: math.e**x
        for h in [1e-1, 1e-5, 1e-7]:
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
        for h in [1e-1, 1e-5, 1e-7]:
            print(
                f"Result for x = 0, h = {h}: {Derivative.second_three_point_central(function, 0, h)}"
            )
            print()
            print(
                f"Result for x = 1, h = {h}: {Derivative.second_three_point_central(function, 1, h)}"
            )
