# just a collection of layers
from numpy import ndarray as Tensor
from lab3.layers import Layer
from typing import Sequence, Iterator, Tuple


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def update(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, gradient: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
