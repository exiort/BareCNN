import numpy as np

from .layer import Layer



class Flatten(Layer):
    a_prev_shape:tuple[int,int,int]
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        n_in, c_in, h_in, w_in = self.a_prev_shape = x.shape 
        # (n, c, h, w) -> (n, c * h * w)
        return x.reshape(n_in, c_in * h_in * w_in)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        #(n, c * h * w) -> (n, c, h, w)
        return grad_output.reshape(self.a_prev_shape)
