from matrix import Matrix
from linear_equations import LinearEquations


class LU:
    @staticmethod
    def get_LU(matrix):
        rows = len(matrix)
        cols = len(matrix[0])

        matrix_clone = Matrix.clone(matrix)

        L = [[0 for _ in range(cols)] for _ in range(rows)]
        U = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                if j > i:
                    tmp = matrix_clone[j][i] / matrix_clone[i][i]

                    for k in range(i, cols + 1):
                        if k != cols:
                            matrix_clone[j][k] = (
                                matrix_clone[j][k] - tmp * matrix_clone[i][k]
                            )

                            if matrix_clone[j][k] == 0:
                                L[j][k] = tmp
                elif j == i:
                    L[i][j] = 1

                U[i][j] = matrix_clone[i][j]

        return (L, U)

    @staticmethod
    def solve(matrix, constant_terms):
        L, U = LU.get_LU(matrix)

        y = LinearEquations.gauss_elimination_pivoting(L, constant_terms)
        result = LinearEquations.gauss_elimination_pivoting(U, y)

        return result

    @staticmethod
    def inverse_matrix(matrix):
        L, U = LU.get_LU(matrix)
        result = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

        for i in range(len(matrix[0])):
            e = [[0 for _ in range(1)] for _ in range(len(matrix))]
            for j in range(len(matrix)):
                e[j][0] = 0

            e[i][0] = 1

            X = LinearEquations.gauss_elimination_pivoting(L, e)
            Y = LinearEquations.gauss_elimination_pivoting(U, X)

            for j in range(len(matrix[0])):
                result[j][i] = Y[j][0]

        return result

    @staticmethod
    def det(matrix):
        L, U = LU.get_LU(matrix)

        L_det = Matrix.laplace_det(L)
        U_det = Matrix.laplace_det(U)

        return L_det * U_det
