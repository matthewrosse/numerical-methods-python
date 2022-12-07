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

    result = Matrix.multiply(matrix_a, matrix_b)

    print()
    print("matrix:")
    Matrix.print(result)


if __name__ == "__main__":
    main()
