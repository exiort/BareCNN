import numpy as np

from barecnn.layers.layer import Layer
from barecnn.loses.loss import Loss
from barecnn.optimizer.optimizer import Optimizer
from barecnn.metrics.metric import Metric


class Model:
    layers:list[Layer]
    loss_fn:Loss
    optimizer:Optimizer
    metrics:list[Metric]
    
    additional_layers:list[Layer]
    
    is_ready:bool
    def __init__(self) -> None:
        self.is_ready = False

    def _forward(self, x:np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def _backward(self, grad_output:np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
    
    def prepare_model(self) -> None:
        if not self.layers:
            return
        if not self.loss_fn:
            return
        if not self.optimizer:
            return 

        self.optimizer.set_parameters([layer.get_parameters() for layer in self.layers])
        self.is_ready = True
        
    def train(self, X_train:np.ndarray, y_train:np.ndarray, epochs:int, batch_size:int) -> None:
        if not self.is_ready:
            return
        
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
        
                prediction = self._forward(X_batch)

                loss = self.loss_fn.forward(prediction, y_batch)
                epoch_loss += loss

                grad_output = self.loss_fn.backward()
                self._backward(grad_output)
                
                self.optimizer.step()
                self.optimizer.zero_grad()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (num_samples / batch_size):.4f}")        

    def evaluate(self, X_test:np.ndarray, y_test:np.ndarray) -> dict[str,float]:
        if not self.is_ready:
            return {}
        
        results = {}

        x = self._forward(X_test)

        results["loss"] = self.loss_fn.forward(x, y_test)

        if not self.metrics:
            return results

        if self.additional_layers:
            for layer in self.additional_layers:
                x = layer.forward(x)

        for metric in self.metrics:
            results[metric.name.lower()] = metric.compute(x, y_test)

        return results
        
    
    def predict(self, x:np.ndarray) -> np.ndarray:
        x = self._forward(x)
        if self.additional_layers:
            for layer in self.additional_layers:
                x = layer.forward(x)

        return x
    
    def save_model(self, filepath:str) -> None:
        param_dict = {}

        for i, layer in enumerate(self.layers):
            if layer.parameters:
                for j, param in enumerate(layer.parameters):
                    key = f"{i}-{j}-{param.param_type.name.lower()}"
                    param_dict[key] = param.value

        np.savez_compressed(filepath, **param_dict)
        print(f"Model is saved to: {filepath}")
        
    def load_model(self, filepath:str) -> None:
        try:
            loaded_params = np.load(filepath)

            for i, layer in enumerate(self.layers):
                if layer.parameters:
                    for j, param in enumerate(layer.parameters):
                        key = f"{i}-{j}-{param.param_type.name.lower()}"
                        layer.load_parameter(loaded_params[key], param.param_type)

            loaded_params.close()
            self.prepare_model()
            print(f"Model is loaded from: {filepath}")
            
        except Exception as e:
            print(e)
    
