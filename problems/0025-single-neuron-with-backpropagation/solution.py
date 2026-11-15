import numpy as np

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    X = np.asarray(features, dtype=float)          # (N, D)
    y = np.asarray(labels, dtype=float)            # (N,)
    w = np.array(initial_weights, dtype=float)     # copy (the "clone().detach()")
    b = float(initial_bias)
    N = len(y)
    mse_values = []

    for _ in range(epochs):
        preds = 1 / (1 + np.exp(-(X @ w + b)))     # forward
        loss = np.mean((preds - y) ** 2)
        mse_values.append(round(loss, 4))

        delta = 2 * (preds - y) / N * preds * (1 - preds)   # dL/dz, shape (N,)
        grad_w = X.T @ delta                                 # (D,)
        grad_b = delta.sum()

        w -= learning_rate * grad_w                          # opt.step()
        b -= learning_rate * grad_b

    return [round(v, 4) for v in w.tolist()], round(b, 4), mse_values