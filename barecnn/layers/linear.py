import numpy as np

from .layer import Layer, LayerType
from barecnn.core.parameter import Parameter, ParameterType
from barecnn.core.param_init import ParamInit, WInitStrategies, BInitStrategies


class LinearLayer(Layer):
    in_features:int
    out_features:int

    W:Parameter
    b:Parameter

    a_prev:np.ndarray
    
    def __init__(self, in_features:int, out_features:int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.W = Parameter(ParameterType.WEIGHT, ParamInit.init_w((out_features, in_features), LayerType.LINEAR, WInitStrategies.HE_NORMAL))
        self.b = Parameter(ParameterType.BIAS, ParamInit.init_b(out_features, BInitStrategies.SMALL_POZITIVE))
        
        self.parameters.extend([self.W, self.b])

    def forward(self, x:np.ndarray) -> np.ndarray:
        self.a_prev = x
        # (out, in) x (n, in).T = (out, n)
        # (out, n).T + (,out) = (n, out)
        return (self.W.value @ x.T).T + self.b.value
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # (n,out).T x (n,in) = (out, in)
        self.W.grad = grad_output.T @ self.a_prev
        # (,out)
        self.b.grad = np.sum(grad_output, axis=0)

        #(n, out) x (out, in) = (n,in)
        return grad_output @ self.W.value

    def load_parameter(self, parameter:np.ndarray, param_type:ParameterType) -> None:
        if param_type == ParameterType.WEIGHT:
            if parameter.shape != (self.out_features, self.in_features):
                raise ValueError()
        
            self.W.value = parameter

        elif param_type == ParameterType.BIAS:
            if parameter.shape != (self.out_features,):
                raise ValueError()

            self.b.value = parameter
