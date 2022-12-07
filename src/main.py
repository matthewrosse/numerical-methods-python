from matrix import Matrix
from complex import Complex
from factorial import factorial
from euler import euler
from linear_equations import LinearEquations


def main():
    # print("Hello, world!")
    # a: Complex = Complex(3, 2)
    # b: Complex = Complex(1, 7)

    # print(Complex.add(a, b))

    # print(a + b)
    # print(a * b)

    # print(factorial(10))
    # print(euler(100))

    # matrix_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # matrix_b = [[2, 3, 4], [5, 7, 8], [9, 2, 3]]
    # matrix_cw2 = [[1, 3, 2], [4, -1, 2], [1, -1, 0]]

    # matrix_5x5 = [
    #     [123, 2, 3, 4, 5],
    #     [6, 7, 8, 9, 10],
    #     [11, 23, 13, 14, 15],
    #     [16, 17, 18, 19, 20],
    #     [21, 22, 23, 24, 27],
    # ]

    # result = Matrix.multiply(matrix_a, matrix_b)

    # print()
    # print("matrix:")
    # Matrix.print(result)

    # print("Det:")
    # print(Matrix.sarrus_det(matrix_b))

    # print()
    # print()
    # print()

    # Matrix.create_minor(matrix_b, 1, 1)

    # print("Laplace det")
    # print()
    # print(Matrix.laplace_det(matrix_5x5))
    # print()
    # print()
    # print()

    # inversed = Matrix.inverse(matrix_cw2)
    # Matrix.print(inversed)
    inverse_matrix_cw1 = [[5, -2, 3], [-2, 3, 1], [-1, 2, 3]]
    constant_terms_cw1 = [[21], [-4], [5]]

    # result_cw1 = LinearEquations.inverse_matrix_method(
    #     inverse_matrix_cw1, constant_terms_cw1
    # )

    # result_cw2 = LinearEquations.cramer_method(inverse_matrix_cw1, constant_terms_cw1)

    # result_cw3 = LinearEquations.gauss_elimination(
    #     inverse_matrix_cw1, constant_terms_cw1
    # )

    # Matrix.print(result_cw3)

    pivoting_matrix = [[0, 1, 1], [1, 1, 1], [2, 0, -1]]
    pivoting_constant_terms = [[1], [2], [0]]

    result_cw5 = LinearEquations.gauss_elimination_pivoting(
        pivoting_matrix, pivoting_constant_terms
    )

    Matrix.print(result_cw5)


if __name__ == "__main__":
    main()
