import math

def softmax_derivative(x: list[float]) -> list[list[float]]:
    # Softmax (stable)
    max_x = max(x)
    exps = [math.exp(z - max_x) for z in x]
    sum_exps = sum(exps)
    s = [e / sum_exps for e in exps]

    # Jacobian
    n = len(s)
    J = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i][j] = s[i] * (1 - s[i])
            else:
                J[i][j] = -s[i] * s[j]
    return J