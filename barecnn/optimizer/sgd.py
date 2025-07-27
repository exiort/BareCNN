import numpy as np

from barecnn.core.parameter import Parameter
from .optimizer import Optimizer



class SGD(Optimizer):
    momentum:float
    v:list[list[np.ndarray]]

    
    def __init__(self, lr:float, momentum:float=0) -> None:
        super().__init__(lr)
        self.momentum = momentum
    
    def set_parameters(self, parameters: list[list[Parameter]]) -> None:
        super().set_parameters(parameters)
        self.v = [[np.zeros_like(param.value) for param in layer] for layer in parameters]

    def step(self) -> None:
        for i, layer in enumerate(self.parameters):
            for j, param in enumerate(layer):
                if self.momentum:
                    self.v[i][j] = self.momentum * self.v[i][j] - self.lr * param.grad
                    param.value += self.v[i][j]

                else:
                    param.value -= self.lr * param.grad

