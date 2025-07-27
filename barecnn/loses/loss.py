import numpy as np 



class Loss:
    y_pred:np.ndarray
    y_true:np.ndarray

    def __init__(self) -> None:...

    def forward(self, y_pred:np.ndarray, y_true:np.ndarray) -> float:...
    
    def backward(self) -> np.ndarray:...
