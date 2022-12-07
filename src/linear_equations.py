from matrix import Matrix
import math


class LinearEquations:
    @staticmethod
    def inverse_matrix_method(matrix, constant_terms):
        inverse_matrix = Matrix.inverse(matrix)

        return Matrix.multiply(inverse_matrix, constant_terms)

    @staticmethod
    def cramer_method(matrix, constant_terms):
        result = [
            [0 for _ in range(len(constant_terms[0]))]
            for _ in range(len(constant_terms))
        ]
        matrix_determinant = Matrix.laplace_det(matrix)
        for k in range(len(matrix[0])):
            # result[k][0] =
            result[k][0] = (
                Matrix.laplace_det(
                    LinearEquations.substitute_matrix_col(matrix, constant_terms, k)
                )
                / matrix_determinant
            )

        return result

    @staticmethod
    def substitute_matrix_col(matrix, constant_terms, col):
        result = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                result[i][j] = matrix[i][j]

        for i in range(len(result)):
            result[i][col] = constant_terms[i][0]

        return result

    @staticmethod
    def gauss_elimination(matrix, constant_terms):
        (
            upper_triangular_matrix,
            upper_triangular_constant_terms,
        ) = Matrix.upper_triangular(matrix, constant_terms)

        result = [
            [0 for _ in range(len(constant_terms[0]))]
            for _ in range(len(constant_terms))
        ]

        n = len(constant_terms)

        for i in range(n - 1, -1, -1):
            result[i][0] = upper_triangular_constant_terms[i][0]

            for j in range(i + 1, n):
                result[i][0] -= upper_triangular_matrix[i][j] * result[j][0]

            result[i][0] = result[i][0] / upper_triangular_matrix[i][i]

        return result

    @staticmethod
    def gauss_elimination_pivoting(matrix, constant_terms):

        result = [
            [0 for _ in range(len(constant_terms[0]))]
            for _ in range(len(constant_terms))
        ]

        matrix_copy = Matrix.clone(matrix)
        constant_terms_copy = Matrix.clone(constant_terms)

        tmp = 0

        for i in range(len(matrix_copy[0])):
            for j in range(i + 1, len(matrix_copy[0])):
                if matrix_copy[i][i] == 0:
                    index = Matrix.find_biggest_column_idx(matrix_copy, i)
                    Matrix.substitute_rows(matrix_copy, index, i)
                    Matrix.substitute_rows(constant_terms_copy, index, i)

                tmp = matrix_copy[j][i] / matrix_copy[i][i]

                for k in range(len(matrix_copy[0]) + 1):
                    if k != len(matrix_copy[0]):
                        matrix_copy[j][k] -= tmp * matrix_copy[i][k]
                    else:
                        constant_terms_copy[j][0] -= tmp * constant_terms_copy[i][0]

        print("test")
        Matrix.print(matrix_copy)
        print()
        Matrix.print(constant_terms_copy)
        print()

        for i in range(len(matrix_copy) - 1, -1, -1):
            tmp = 0

            for j in range(i, len(matrix_copy[0])):
                tmp += matrix_copy[i][j] * result[j][0]

            result[i][0] = (constant_terms_copy[i][0] - tmp) / matrix_copy[i][i]

        return result

    @staticmethod
    def gauss_jordan(matrix, constant_terms):
        diagonal_matrix, diagonal_constant_terms = Matrix.diagonal(
            matrix, constant_terms
        )

        result = [
            [0 for _ in range(len(constant_terms[0]))]
            for _ in range(len(constant_terms))
        ]

        for k in range(len(diagonal_matrix)):
            result[k][0] = diagonal_constant_terms[k][0] / diagonal_matrix[k][k]

        return result

    @staticmethod
    def matrix_inverse_gauss_method(matrix):
        rows = len(matrix)
        cols = len(matrix[0])

        # exception when rows != cols
        if rows != cols:
            raise Exception("Wrong matrix dimensions.")

        result = [[0 for _ in range(cols)] for _ in range(rows)]
        e = [[0 for _ in range(1)] for _ in range(rows)]

        for i in range(rows):
            for j in range(rows):
                if j == i:
                    e[j][0] = 1
                else:
                    e[j][0] = 0

            gauss_result = LinearEquations.gauss_elimination_pivoting(matrix, e)

            for j in range(rows):
                result[j][i] = gauss_result[j][0]

        return result
