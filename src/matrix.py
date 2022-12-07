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
            raise Exception(
                "Matrices cannot be multiplied, incorrect dimensions.")

        result = [[0 for row in range(cols_b)] for col in range(rows_a)]

        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]

        return result

    @staticmethod
    def sarrus_det_2x2(matrix):
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    @staticmethod
    def sarrus_det_3x3(matrix):
        # need to check dimensions and throw error
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
    def create_minor(matrix, row, col):
        result_rows = len(matrix) - 1
        result_cols = len(matrix[0]) - 1

        result = [[0 for row in range(result_cols)]
                  for col in range(result_rows)]

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
    def sarrus_det(matrix):
        if len(matrix) == 2 and len(matrix[0]) == 2:
            return Matrix.sarrus_det_2x2(matrix)

        if len(matrix) == 3 and len(matrix[0]) == 3:
            return Matrix.sarrus_det_3x3(matrix)

        raise Exception(
            "Incorrect dimensions! You need to provide 2x2 or 3x3 matrix.")

    @staticmethod
    def print(matrix) -> None:
        for row in matrix:
            for col in row:
                print(f"{col:.2f}", end="\t")
            print()
