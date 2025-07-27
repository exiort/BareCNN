import numpy as np

from barecnn.core.parameter import Parameter, ParameterType
from barecnn.core.math import Math
from barecnn.core.param_init import ParamInit, WInitStrategies, BInitStrategies
from .layer import Layer, LayerType



class ConvolutionalLayer(Layer):
    batch_size:int
    in_channels:int
    out_channels:int
    kernel_size:int
    stride:int

    kernels:Parameter

    a_prev:np.ndarray
    
    
    def __init__(self, in_channel:int, out_channel:int, kernel_size:int, stride:int, w_init_strategy:WInitStrategies=WInitStrategies.HE_NORMAL, b_init_strategy:BInitStrategies = BInitStrategies.SMALL_POZITIVE) -> None:
        super().__init__()

        #nchw
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.kernels = Parameter(ParameterType.WEIGHT, ParamInit.init_w((out_channel, in_channel, kernel_size, kernel_size), LayerType.CONV2D, w_init_strategy)) 
        self.biases = Parameter(ParameterType.BIAS, ParamInit.init_b(out_channel, b_init_strategy))
        
        self.parameters.extend([self.kernels, self.biases])
        
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.a_prev = x
        self.batch_size, _, h_in, w_in = x.shape
        h_out, w_out = (h_in - self.kernel_size) // self.stride + 1, (w_in - self.kernel_size) // self.stride + 1

        #(N * h_out * w_out, c_in * kernel_size * kernel_size)
        tensor_matrix = Math.im2col(x, (self.kernel_size, self.kernel_size), self.stride)
        #(c_out * 1 * 1, kernel_size * kernel_size * c_in).T
        kernel_matrix = Math.im2col(self.kernels.value, (self.kernel_size, self.kernel_size), 1).T
        
        #(N * h_out * w_out, c_out) -> (N, h_out, w_out, c_out) -> (N, c_out, h_out, w_out)
        return (tensor_matrix @ kernel_matrix + self.biases.value).reshape(self.batch_size, h_out, w_out, self.out_channels).transpose(0, 3, 1, 2)    

    def backward(self, grad_output:np.ndarray) -> np.ndarray:
        _, _, h_in, w_in = self.a_prev.shape

        #(N * h_out * w_out, c_in * kernel_size * kernel_size)
        prev_matrix = Math.im2col(self.a_prev, (self.kernel_size, self.kernel_size), self.stride)
        #(N, c_out, h_out, w_out) -> (N, h_out, w_out, c_out) -> (N * h_out * w_out, c_out) 
        grad_matrix = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        #(c_in * kernel_size * kernel_size, c_out)
        w_grads = prev_matrix.T @ grad_matrix
        self.kernels.grad = w_grads.T.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
            
        self.biases.grad = np.sum(grad_matrix, axis=0)

        #(c_in * kernel_size * kernel_size, c_out)
        kernel_matrix = Math.im2col(self.kernels.value, (self.kernel_size, self.kernel_size), 1).T
        prev_grad_matrix = grad_matrix @ kernel_matrix.T
        
        return Math.col2im(
            prev_grad_matrix,
            (self.batch_size, self.in_channels, h_in, w_in),
            (self.kernel_size, self.kernel_size),
            self.stride
        )
        
    def load_parameter(self, parameter:np.ndarray, param_type:ParameterType) -> None:
        if param_type == ParameterType.WEIGHT:
            if parameter.shape != (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size):
                raise ValueError()

            self.kernels.value = parameter
            
        elif param_type == ParameterType.BIAS:
            if parameter.shape != (self.out_channels,):
                raise ValueError()

            self.biases.value = parameter
        
