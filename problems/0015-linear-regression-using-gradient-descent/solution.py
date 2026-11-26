
import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:

    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))

    for i in range(iterations):
        # Predictions
        y_pred = X @ theta

        # Error
        error = y_pred - y

        # Gradient
        gradient = (1 / m) * (X.T @ error)

        # Update weights
        theta = theta - alpha * gradient

    return theta.flatten()
