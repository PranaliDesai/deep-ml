import numpy as np
import math

def log_softmax(scores: list) -> np.ndarray:
    max_z = max(scores)
    scores = [z - max_z for z in scores]
    sum_ez = sum([math.e**z for z in scores])
    log_sum = np.log(sum_ez)
    return [z - log_sum for z in scores]