import math

def softmax(scores: list[float]) -> list[float]:
    max_z = max(scores)
    scores = [z - max_z for z in scores]
    sum_ez = sum([math.e**z for z in scores])
    return [math.e**z / sum_ez for z in scores]