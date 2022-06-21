from typing import List
from Layer import Layer

class NeuralNetwork:
    """
    Neural Network main class
    """
    def __init__(self, layers: List[Layer]):
        self.layers = layers
    def feed_forward(self, inputs: List[float]) -> List[float]:
        """
        Feeds forward each layer in the network
        """
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return inputs

network = NeuralNetwork([Layer(4, 3, lambda x: x)])
print(network.feed_forward([1, 2, 3, 4]))
