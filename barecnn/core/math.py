import numpy as np
from numpy.lib.stride_tricks import sliding_window_view



class Math:
    @staticmethod
    def im2col(
        image: np.ndarray, # NCHW
        kernel_shape: tuple[int, int], #HW
        stride: int|tuple[int, int]
    ) -> np.ndarray:
        h_k, w_k = kernel_shape
        _, c_in, _, _ = image.shape
        
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        windows = sliding_window_view(
            image,
            (1, c_in, h_k, w_k)
        )[::, ::, ::stride_h, ::stride_w]

        return windows.reshape(-1, c_in * h_k * w_k)

    @staticmethod
    def col2im(
        col_matrix: np.ndarray,
        input_shape: tuple[int, int, int, int],
        kernel_shape: tuple[int, int],
        stride: int|tuple[int, int]
    ) -> np.ndarray:
        n_in, c_in, h_in, w_in = input_shape
        h_k, w_k = kernel_shape

        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        h_out, w_out = (h_in - h_k) // stride_h + 1, (w_in - w_k) // stride_w + 1

        image = np.zeros((n_in, c_in, h_in, w_in))
        
        col_reshaped = col_matrix.reshape(n_in, 1, h_out, w_out, c_in, h_k, w_k)

        for i in range(n_in):
            for j in range(h_out):
                for k in range(w_out):
                    h_start = j * stride_h
                    h_end = h_start + h_k
                    w_start = k * stride_w
                    w_end = w_start + w_k

                    image[i, :, h_start:h_end, w_start:w_end] += col_reshaped[i, 0, j, k]
                    
        return image
        
