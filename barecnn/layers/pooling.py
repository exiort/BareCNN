import numpy as np
from enum import Enum

from .layer import Layer
from barecnn.core.math import Math



class PoolingStrategies(Enum):
    MAX = 1
    AVERAGE = 2



class PoolingLayer(Layer):
    in_channels:int
    kernel_size:int
    stride:int
    strategy:PoolingStrategies
    
    a_prev_shape:tuple[int,int,int,int]
    max_indices:np.ndarray
    
    def __init__(self, in_channels:int, kernel_size:int, stride:int, strategy:PoolingStrategies) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.strategy = strategy
        
    def forward(self, x:np.ndarray) -> np.ndarray:
        n_in, c_in, h_in, w_in = self.a_prev_shape = x.shape
        h_out, w_out = (h_in - self.kernel_size) // self.stride + 1, (w_in - self.kernel_size) // self.stride + 1

        # (n_in, c_in, h_in, w_in) -> (n_in * h_out * w_out, c_in * kernel_size, kernel_size)
        tensor_matrix = Math.im2col(x, (self.kernel_size, self.kernel_size), self.stride)
        # (n_in * h_out * w_out, c_in * kernel_size * kernel_size) -> (n_in * h_out * w_out, c_in, kernel_size * kernel_size)
        tensor_m = tensor_matrix.reshape(n_in * h_out * w_out, c_in, self.kernel_size * self.kernel_size)

        if self.strategy == PoolingStrategies.MAX:
            # (n_in * h_out * w_out, c_in, kernel_size * kernel_size) -> (n_in * h_out * w_out, c_in)
            self.max_indices = np.argmax(tensor_m, axis=2)
            # (n_in * h_out * w_out, c_in, kernel_size * kernel_size) -> (n_in * h_out * w_out, c_in)
            result = np.max(tensor_m, axis=2)

        elif self.strategy == PoolingStrategies.AVERAGE:
            # (n_in * h_out * w_out, c_in, kernel_size * kernel_size) -> (n_in * h_out * w_out, c_in)
            result = np.mean(tensor_m, axis=2)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.strategy}")

        #(n_in * h_out * w_out, c_in) -> (n_in, h_out, w_out, c_in)
        result = result.reshape((n_in, h_out, w_out, c_in))
        #(n_in, h_out, w_out, c_in) -> (n_in, c_in, h_out, w_out)
        return result.transpose(0, 3, 1, 2)

    def backward(self, grad_output:np.ndarray) -> np.ndarray:
        n_in, c_in, h_out, w_out = grad_output.shape
        patch_area = self.kernel_size * self.kernel_size
        effective_batch = n_in * h_out * w_out
        
        # (n_in, c_in, h_out, w_out) -> (n_in, h_out, w_out, c_in)
        grad_t = grad_output.transpose(0, 2, 3, 1)
        
        #(n_in, h_out, w_out, c_in) -> (n_in * h_out * w_out,  c_in)
        grad_m = grad_t.reshape(effective_batch, c_in)

        if self.strategy == PoolingStrategies.MAX:
            # (n_in * h_out * w_out, c_in, kernel_size * kernel_size)
            grad_patches = np.zeros((effective_batch, c_in, patch_area))
            # (n_in * h_out * w_out, c_in, kernel_size * kernel_size)
            np.put_along_axis(grad_patches, self.max_indices[:,:,None], grad_m[:,:,None], axis=2)

        elif self.strategy == PoolingStrategies.AVERAGE:
            # (n_in * h_out * w_out, c_in, kernel_size * kernel_size)
            grad_patches = np.repeat(grad_m[:,:,None], patch_area, axis=2) / patch_area
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.strategy}")

        # (n_in * h_out * w_out, c_in * kernel_size * kernel_size) -> (n_in * h_out * w_out, c_in * kernel_size * kernel_size)
        grad_patches = grad_patches.reshape(effective_batch, c_in * patch_area)

        # (n_in * h_out * w_out, c_in * kernel_size * kernel_size) -> (n_in, c_in, h_out, w_out)
        return Math.col2im(
            grad_patches,
            self.a_prev_shape,
            (self.kernel_size, self.kernel_size),
            self.stride
        )
