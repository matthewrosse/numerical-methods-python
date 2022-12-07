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
