import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q,K,V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    row_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - row_max)
    soft_max = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return soft_max @ V


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    # Q, K, V: (4, 8)
    q_heads = np.split(Q, n_heads, axis=1)  # list of 2 × (4, 4)
    k_heads = np.split(K, n_heads, axis=1)  # list of 2 × (4, 4)
    v_heads = np.split(V, n_heads, axis=1)  # list of 2 × (4, 4)

    outputs = []
    for i in range(n_heads):
        # inside: scores (4, 4) → weights (4, 4) → @ V_i → (4, 4)
        outputs.append(self_attention(q_heads[i], k_heads[i], v_heads[i]))  # (4, 4)

    return np.concatenate(outputs, axis=1)  # (4, 8)
