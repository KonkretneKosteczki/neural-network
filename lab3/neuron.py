from typing import Dict, Callable
from numpy import ndarray as Tensor
import numpy as np

F = Callable[[Tensor], Tensor]


class Neuron:
    def __init__(self, input_size: int, weights=None, bias=None) -> None:
        self.temp_input_size = input_size
        self.weights = weights if weights is not None else np.random.randn(input_size)
        self.bias = bias if bias is not None else np.random.randn()
        self.inputs = Tensor([])

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.weights + self.bias

    def backward(self, gradient: Tensor) -> Tensor:
        # self.grads["b"] = np.sum(gradient, axis=0)
        # self.grads["w"] = self.inputs.T @ gradient
        return gradient @ self.inputs.T @ gradient.T


class Layer:
    def __init__(self) -> None:
        self.grads: Dict[str, Tensor] = {}
        self.params: Dict[str, Tensor] = {}
        self.neurons: Tensor = Tensor([])

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, gradient: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size: int, neurons_nr: int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(input_size, neurons_nr)
        self.params["b"] = np.random.randn(neurons_nr)
        self.neurons = [Neuron(input_size, self.params["w"][:, i], self.params["b"][i]) for i
                        in range(neurons_nr)]
        # self.neurons = np.array([Neuron(input_size) for i in range(output_size)])
        self.inputs = Tensor([])

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return np.array([neuron.forward(inputs) for neuron in self.neurons]).T

    def backward(self, gradient: Tensor) -> Tensor:
        self.grads["b"] = np.sum(gradient, axis=0)
        self.grads["w"] = self.inputs.T @ gradient
        return gradient @ np.array([neuron.weights for neuron in self.neurons])
