import numpy as np

def pos_encoding(position: int, d_model: int):
    if position == 0 or d_model <= 0:
        return -1

    pe = np.zeros((position, d_model))

    pos = np.arange(position)[:, np.newaxis]
    i = np.arange(0, d_model, 2)

    angle = pos / (10000 ** (i / d_model))

    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)

    return pe.astype(np.float16)