from matrix import Matrix
from complex import Complex
from factorial import factorial
from euler import euler


def main():
    print("Hello, world!")
    a: Complex = Complex(3, 2)
    b: Complex = Complex(1, 7)

    print(Complex.add(a, b))

    print(a + b)
    print(a * b)

    print(factorial(10))
    print(euler(100))

    matrix_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matrix_b = [[2, 3, 4], [5, 7, 8], [9, 2, 3]]

    matrix_5x5 = [
        [123, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 23, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 27],
    ]

    result = Matrix.multiply(matrix_a, matrix_b)

    print()
    print("matrix:")
    Matrix.print(result)

    print("Det:")
    print(Matrix.sarrus_det(matrix_b))

    print()
    print()
    print()

    Matrix.create_minor(matrix_b, 1, 1)

    print("Laplace det")
    print()
    print(Matrix.laplace_det(matrix_5x5))


if __name__ == "__main__":
    main()
