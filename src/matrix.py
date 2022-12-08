import random as rd
import math


class Matrix:
    @staticmethod
    def multiply(a, b):
        rows_a = len(a)
        cols_a = len(a[0])
        rows_b = len(b)
        cols_b = len(b[0])

        if cols_a != rows_b:
            raise Exception("Matrices cannot be multiplied, incorrect dimensions.")

        result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]

        return result

    @staticmethod
    def is_square_matrix(matrix) -> bool:
        if len(matrix) == len(matrix[0]):
            return True

        return False

    @staticmethod
    def sarrus_det_2x2(matrix):
        if not Matrix.is_square_matrix(matrix):
            raise Exception("Matrix must be square matrix to perform this operation.")

        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    @staticmethod
    def sarrus_det_3x3(matrix):
        if not Matrix.is_square_matrix(matrix):
            raise Exception("Matrix must be square matrix to perform this operation.")

        first = (
            matrix[0][0] * matrix[1][1] * matrix[2][2]
            + matrix[1][0] * matrix[2][1] * matrix[0][2]
            + matrix[2][0] * matrix[0][1] * matrix[1][2]
        )
        second = (
            matrix[1][0] * matrix[0][1] * matrix[2][2]
            + matrix[0][0] * matrix[2][1] * matrix[1][2]
            + matrix[2][0] * matrix[1][1] * matrix[0][2]
        )

        return first - second

    @staticmethod
    def sarrus_det(matrix):
        if not Matrix.is_square_matrix(matrix):
            raise Exception("Matrix must be square matrix to perform this operation.")

        if len(matrix) == 2 and len(matrix[0]) == 2:
            return Matrix.sarrus_det_2x2(matrix)

        if len(matrix) == 3 and len(matrix[0]) == 3:
            return Matrix.sarrus_det_3x3(matrix)

        raise Exception("Incorrect dimensions! You need to provide 2x2 or 3x3 matrix.")

    @staticmethod
    def create_minor(matrix, row, col):
        result_rows = len(matrix) - 1
        result_cols = len(matrix[0]) - 1

        result = [[0 for _ in range(result_cols)] for _ in range(result_rows)]

        for i in range(len(matrix)):
            minor_row = i
            if i > row:
                minor_row -= 1

            for j in range(len(matrix[0])):
                minor_col = j
                if j > col:
                    minor_col -= 1

                if i != row and j != col:
                    result[minor_row][minor_col] = matrix[i][j]

        return result

    @staticmethod
    def laplace_det(matrix):
        if len(matrix) < 4:
            return Matrix.sarrus_det(matrix)

        rand_row = rd.randrange(0, len(matrix))

        result = 0

        for i in range(len(matrix[0])):
            sub_matrix = Matrix.create_minor(matrix, rand_row, i)

            result += (
                matrix[rand_row][i]
                * math.pow(-1, rand_row + i)
                * Matrix.laplace_det(sub_matrix)
            )

        return result

    @staticmethod
    def inverse(matrix):
        if not Matrix.is_square_matrix(matrix):
            raise Exception("Matrix must be square matrix to perform this operation.")

        result = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

        root_matrix_det = Matrix.laplace_det(matrix)
        root_matrix_tranposed = Matrix.transpose(matrix)

        for i in range(len(result)):
            for j in range(len(result[0])):
                result[i][j] = (
                    (1 / root_matrix_det)
                    * math.pow(-1, i + j)
                    * Matrix.laplace_det(
                        Matrix.create_minor(root_matrix_tranposed, i, j)
                    )
                )
        return result

    @staticmethod
    def transpose(matrix):
        return [
            [matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))
        ]

    @staticmethod
    def clone(matrix):
        return [
            [matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))
        ]

    @staticmethod
    def upper_triangular(matrix, constant_terms):
        if not Matrix.is_square_matrix(matrix):
            raise Exception("Matrix must be square matrix to perform this operation.")

        cloned_matrix = Matrix.clone(matrix)
        cloned_constant_terms = Matrix.clone(constant_terms)

        n = len(cloned_matrix)

        multiplier = 0

        for i in range(n):
            for j in range(n):
                if j > i:
                    multiplier = cloned_matrix[j][i] / cloned_matrix[i][i]

                    for k in range(i, n + 1):
                        if k != n:
                            cloned_matrix[j][k] = (
                                cloned_matrix[j][k] - multiplier * cloned_matrix[i][k]
                            )
                        else:
                            cloned_constant_terms[j][0] = (
                                cloned_constant_terms[j][0]
                                - multiplier * cloned_constant_terms[i][0]
                            )

        return (cloned_matrix, cloned_constant_terms)

    @staticmethod
    def diagonal(matrix, constant_terms):
        if not Matrix.is_square_matrix(matrix):
            raise Exception("Matrix must be square matrix to perform this operation.")

        cloned_matrix = Matrix.clone(matrix)
        cloned_constant_terms = Matrix.clone(constant_terms)

        n = len(cloned_matrix)

        for i in range(n):
            for j in range(n):
                if j != i:
                    tmp = cloned_matrix[j][i] / cloned_matrix[i][i]

                    for k in range(i, n + 1):
                        if k != n:
                            cloned_matrix[j][k] -= tmp * cloned_matrix[i][k]
                        else:
                            cloned_constant_terms[j][0] -= (
                                tmp * cloned_constant_terms[i][0]
                            )

        return (cloned_matrix, cloned_constant_terms)

    @staticmethod
    def find_biggest_column_idx(matrix, col):
        biggest_idx = 0
        for i in range(1, len(matrix)):
            if matrix[i][col] > matrix[i - 1][col]:
                biggest_idx = i

        return biggest_idx

    @staticmethod
    def substitute_rows(matrix, first_row, second_row):
        for i in range(len(matrix[0])):
            tmp = matrix[first_row][i]
            matrix[first_row][i] = matrix[second_row][i]
            matrix[second_row][i] = tmp

    @staticmethod
    def print(matrix, decimal_places=2) -> None:
        for row in matrix:
            for col in row:
                print(f"{col:.{decimal_places}f}", end="\t")
            print()
