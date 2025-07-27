import numpy as np

from .layer import Layer



class ReLu(Layer):
    z_prev:np.ndarray

    
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.z_prev = x
        return np.maximum(0, x) 

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # dL/da * da/dz = grad_output *   = dL/dz
        return grad_output * (self.z_prev > 0) 


    
class Sigmoid(Layer):
    a_prev:np.ndarray

    
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.a_prev = 1 / (1 + np.exp(-x))
        return self.a_prev

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # dL/da * da/dz = grad_output * sigmoid'(z) = dL/dz
        #sigmoid'(z) = a * (1-a)
        return grad_output * self.a_prev * (1 - self.a_prev)


    
class TanH(Layer):
    a_prev:np.ndarray

    
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.a_prev = np.tanh(x) 
        return self.a_prev

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        #tanh'(z) = 1 - a^2
        return grad_output * (1 - self.a_prev ** 2)



class Softmax(Layer):
    a_prev:np.ndarray


    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
