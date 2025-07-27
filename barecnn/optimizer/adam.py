import numpy as np

from barecnn.core.parameter import Parameter
from .optimizer import Optimizer



class Adam(Optimizer):
    beta1:float
    beta2:float
    eps:float
    t:int

    m:list[list[np.ndarray]]
    v:list[list[np.ndarray]]

    
    def __init__(self, lr: float, beta1:float=0.9, beta2:float=0.999, eps:float=1e-8) -> None:
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
    def set_parameters(self, parameters: list[list[Parameter]]) -> None:
        super().set_parameters(parameters)
        self.m = [[np.zeros_like(param.value) for param in layer] for layer in parameters]
        self.v = [[np.zeros_like(param.value) for param in layer] for layer in parameters]
 
    def step(self) -> None:
        self.t += 1

        for i, layer in enumerate(self.parameters):
            for j, param in enumerate(layer):
                self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * param.grad
                self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * (param.grad ** 2)

                m_hat = self.m[i][j] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][j] / (1 - self.beta2 ** self.t)

                param.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
