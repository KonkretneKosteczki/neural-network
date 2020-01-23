# each layer passes inputs forward and propagate gradient backward
from typing import Dict, Callable
from numpy import ndarray as Tensor
import numpy as np

F = Callable[[Tensor], Tensor]


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, gradient: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    """
    output = inputs @ w + bias
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        # input = batch_size x input_size
        # output = batch_size x output_size
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        # output = input @ w + b
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, gradient: Tensor) -> Tensor:
        # single variable calculus
        #  for y=f(x) and x=a*b+c
        #  dy/da = f'(x) * b
        #  dy/db = f'(x) * a
        #  dy/dc = f'(x)

        #  for y=f(x) and x=a@b+c
        #  dy/da = f'(x) @ b.T
        #  dy/db = a.T @ f'(x)
        #  dy/dc = f'(x)

        self.grads["b"] = np.sum(gradient, axis=0)
        self.grads["w"] = self.inputs.T @ gradient

        return gradient @ self.params["w"].T


class Activation(Layer):
    # applies function elementwise to its inputs

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, gradient: Tensor) -> Tensor:
        # gradient in respect to input (just chain rule)
        return self.f_prime(self.inputs) * gradient


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)
