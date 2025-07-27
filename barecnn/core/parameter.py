import numpy as np
from enum import Enum



class ParameterType(Enum):
    WEIGHT = 1
    BIAS = 2

    
    
class Parameter:    
    value:np.ndarray
    grad:np.ndarray
    param_type:ParameterType

    
    def __init__(self, param_type:ParameterType, value:np.ndarray, grad:np.ndarray|None=None) -> None:
        self.param_type = param_type
        self.value = value
        if grad is None:
            grad = np.zeros_like(value)
        self.grad = grad
            
    def __repr__(self) -> str:
        return self.param_type.name + ": " + "\nValue:" + self.value.__repr__()# + "\nGrad:" + self.grad.__repr__()
    
    
    
