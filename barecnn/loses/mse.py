import numpy as np

from .loss import Loss



class MSELoss(Loss):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # (n, in)
        self.y_pred = y_pred
        # (n, in)
        self.y_true = y_true

        return float(np.mean((y_true - y_pred) ** 2))        
        
    def backward(self) -> np.ndarray:
        return (2 / self.y_pred.size) * (self.y_pred - self.y_true)
