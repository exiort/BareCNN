import numpy as np

from .loss import Loss



class SoftmaxCrossEntropyLoss(Loss):
    class_weights:np.ndarray|None
    
    def __init__(self, class_weights:np.ndarray|None=None) -> None:
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        #(n, in)
        self.y_true = y_true

        # (n, in) - (n, 1) -> (n, in)
        shifted_y_pred = y_pred - np.max(y_pred, axis=1, keepdims=True)
        # (n, in) -> (n, in)
        exp_scores = np.exp(shifted_y_pred)

        # (n, in) / (n, 1) -> (n, in)
        self.y_pred = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        per_sample_loss = -np.sum(self.y_true * np.log(self.y_pred + 1e-15), axis=1)        

        if self.class_weights is not None:
            # (n, out) -> (n,)
            true_class_idx = np.argmax(self.y_true, axis=1)
            #(n,)
            weights_for_samples = self.class_weights[true_class_idx]
            #(n,)
            per_sample_loss = per_sample_loss * weights_for_samples

        return float(np.mean(per_sample_loss))

    def backward(self) -> np.ndarray:
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]
        
