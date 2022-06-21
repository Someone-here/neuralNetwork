from types import FunctionType
from typing import List
import numpy as np

np.random.seed(0)

class Layer:
    """
    A Layer class containing neurons(weights, baises, activation)
    """
    def __init__(self, n_inputs: int, n_neurons: int, activation: FunctionType):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.baises = np.random.randn(1, n_neurons)
        self.activation = activation
    def feed_forward(self, inputs: List):
        """
        computes the weights, baises and inputs along with activation
        """
        return self.activation(np.add(np.dot(inputs, self.weights), self.baises))

