import numpy as np
from enum import Enum

from barecnn.layers.layer import LayerType



class WInitStrategies(Enum):
    HE_NORMAL = 1
    XAVIER_NORMAL = 2
    
    HE_UNIFORM = 3    
    XAVIER_UNIFORM = 4
    
    
    
class BInitStrategies(Enum):
    ZEROS = 1
    SMALL_POZITIVE = 2

    
    
class ParamInit:

    @staticmethod
    def _calc_fan(shape:tuple, layer_type:LayerType) -> tuple[int, int]:
        if layer_type == LayerType.LINEAR:
            fan_in = shape[1]
            fan_out = shape[0]

        elif layer_type == LayerType.CONV2D:
            # (out, in, kernel_h, kernel_w)
            filt_area = shape[2] * shape[3]
            fan_in = shape[1] * filt_area
            fan_out = shape[0] * filt_area
            
        else:
            raise ValueError()
            
        return fan_in, fan_out
    
    
    @staticmethod
    def init_w(shape:tuple,  layer_type:LayerType, strategy:WInitStrategies) -> np.ndarray:
        fan_in, fan_out = ParamInit._calc_fan(shape, layer_type)
        
        if strategy == WInitStrategies.HE_NORMAL:
            std = np.sqrt(2.0 / fan_in)
            return np.random.normal(0, std, shape)

        elif strategy == WInitStrategies.XAVIER_NORMAL:
            std = np.sqrt(2 / (fan_in + fan_out))
            return np.random.normal(0, std, shape)

        elif strategy == WInitStrategies.HE_UNIFORM:
            limit = np.sqrt(6 / fan_in)
            return np.random.uniform(-limit, limit, shape)

        elif strategy == WInitStrategies.XAVIER_UNIFORM:
            limit = np.sqrt(6 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, shape)
    
    @staticmethod
    def init_b(size:int, strategy:BInitStrategies) -> np.ndarray:
        if strategy == BInitStrategies.ZEROS:
            return np.zeros(size)

        elif strategy == BInitStrategies.SMALL_POZITIVE:
            return np.full(size, 0.01)
    


