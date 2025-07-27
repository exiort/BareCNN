import numpy as np
from enum import Enum 

from barecnn.core.parameter import Parameter, ParameterType



class LayerType(Enum):
    ACTIVATION = 1
    CONV2D = 2
    POOLING = 3
    LINEAR = 4
    FLATTEN = 5
    
    
    

class Layer:
    parameters:list[Parameter]

    
    def __init__(self) -> None:
        self.parameters = []

    def forward(self, x:np.ndarray) -> np.ndarray:...

    def backward(self, grad_output:np.ndarray) -> np.ndarray:...

    def get_parameters(self) -> list[Parameter]:
        return list(self.parameters)

    def load_parameter(self, parameter:np.ndarray, param_type:ParameterType) -> None:...
