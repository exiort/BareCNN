import numpy as np



class Metric:

    def __init__(self) -> None:
        pass
    
    def compute(self, y_preds:np.ndarray, y_trues:np.ndarray) -> float:...

    @property
    def name(self) -> str:...
