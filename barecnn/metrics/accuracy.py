import numpy as np

from .metric import Metric



class Accuracy(Metric):

    def __init__(self) -> None:
        super().__init__()

    def compute(self, y_preds: np.ndarray, y_trues: np.ndarray) -> float:
        #y_pred -> probability dist, y_true -> one_hot encoded

        #TruePoz / Total
        predicted = np.argmax(y_preds, axis=1)
        trues = np.argmax(y_trues, axis=1)

        correct_pred = (predicted == trues).sum()
        return correct_pred / y_preds.shape[0]

    @property
    def name(self) -> str:
        return "Accuracy"
