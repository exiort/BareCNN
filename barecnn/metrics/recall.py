import numpy as np

from .metric import Metric



class Recall(Metric):

    def __init__(self) -> None:
        super().__init__()

    def compute(self, y_preds: np.ndarray, y_trues: np.ndarray) -> float:
        #y_pred -> probability dist, y_true -> one_hot encoded
        
        # TruePoz / (TruePoz + FalseNeg)
        predicted = np.argmax(y_preds, axis=1)
        trues = np.argmax(y_trues, axis=1)

        per_class_recall:list[float] = []

        for c in range(y_preds.shape[1]):
            #true_positive
            tp = np.sum((predicted == c) & (trues == c))

            #false_negative
            fn = np.sum((predicted != c) & (trues == c))

            if (tp + fn) == 0:
                per_class_recall.append(0)

            else:
                per_class_recall.append(tp / (tp + fn))

        return float(np.mean(per_class_recall))

    @property
    def name(self) -> str:
        return "Recall"
