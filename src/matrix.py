class Matrix:
    @staticmethod
    def multiply(a, b):
        rows_a = len(a)
        cols_a = len(a[0])
        rows_b = len(b)
        cols_b = len(b[0])

        if cols_a != rows_b:
            raise Exception("Matrices cannot be multiplied, incorrect dimensions.")

        result = [[0 for row in range(cols_b)] for col in range(rows_a)]

        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]

        return result

    @staticmethod
    def print(matrix) -> None:
        for row in matrix:
            for col in row:
                print(f"{col:.2f}", end="")
            print()
