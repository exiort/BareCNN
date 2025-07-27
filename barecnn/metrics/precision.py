import numpy as np

from .metric import Metric



class Precision(Metric):

    def __init__(self) -> None:
        super().__init__()

    def compute(self, y_preds: np.ndarray, y_trues: np.ndarray) -> float:
        #y_pred -> probability dist, y_true -> one_hot encoded

        # TruePoz / (TruePoz + FalsePoz)
        predicted = np.argmax(y_preds, axis=1)
        trues = np.argmax(y_trues, axis=1)
    
        per_class_precision:list[float] = []

        for c in range(y_preds.shape[1]):
            #true_positive
            tp = np.sum((predicted == c) & (trues == c))

            #false_positive
            fp = np.sum((predicted == c) & (trues != c))

            if (tp + fp) == 0:
                per_class_precision.append(0)

            else:
                per_class_precision.append(tp / (tp + fp))

        return float(np.mean(per_class_precision))

    @property
    def name(self) -> str:
        return "Precision"
