from linear_equations import LinearEquations


class Interpolation:
    @staticmethod
    def solve(nodes, base_functions, values):
        n = len(nodes)
        base = [[base_functions[j](nodes[i]) for j in range(n)] for i in range(n)]

        result = LinearEquations.gauss_elimination_pivoting(base, values)

        return result
