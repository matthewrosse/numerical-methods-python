from factorial import factorial


def euler(n: int) -> float:
    result = 0

    for k in range(0, n):
        result += 1 / (factorial(k))

    return result
