from typing import List, Dict, Any

class LRPStrategy:
    def __init__(self, layers: List[Dict[str, Any]] = None,
                 pooling: List[Dict[str, Any]] = None):
        self.layers = layers
        self.pooling = pooling