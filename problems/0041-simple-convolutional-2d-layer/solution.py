import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # 1. Pad
    padded = np.pad(input_matrix, padding, mode='constant')

    # 2. Output size
    out_h = (input_height + 2 * padding - kernel_height) // stride + 1
    out_w = (input_width + 2 * padding - kernel_width) // stride + 1

    # 3. Allocate
    output_matrix = np.zeros((out_h, out_w))

    # 4. Slide the kernel
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            window = padded[r:r + kernel_height, c:c + kernel_width]
            output_matrix[i, j] = np.sum(window * kernel)

    return output_matrix