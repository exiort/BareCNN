import numpy as np

from barecnn.core.parameter import Parameter



class Optimizer:
    parameters:list[list[Parameter]]
    lr:float
    
    
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def set_parameters(self, parameters:list[list[Parameter]]) -> None:
        self.parameters = parameters
        
    def step(self) -> None:...

    def zero_grad(self):
        for layer in self.parameters:
            for param in layer:
                param.grad = np.zeros_like(param.value)

            
