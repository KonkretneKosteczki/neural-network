import numpy as np
from lab3.neuron import Layer
from numpy import ndarray as Tensor
from typing import Callable

F = Callable[[Tensor], Tensor]


class Activation(Layer):
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


class Tanh(Activation):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_prime)

    @staticmethod
    def tanh(x: Tensor) -> Tensor:
        return np.tanh(x)

    def tanh_prime(self, x: Tensor) -> Tensor:
        y = self.tanh(x)
        return 1 - y ** 2


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)

    @staticmethod
    def sigmoid(x: Tensor) -> Tensor:
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x: Tensor) -> Tensor:
        sigmoid = self.sigmoid(x)
        return sigmoid * (1 - sigmoid)


class HeavisideStep(Activation):
    def __init__(self):
        super().__init__(self.heaviside_step, self.heaviside_prime)

    @staticmethod
    def heaviside_step(x: Tensor) -> Tensor:
        # print(np.array([1 if value > 0 else 0 for value in x]))
        return np.heaviside(x, 0)

    @staticmethod
    def heaviside_prime(x: Tensor) -> Tensor:
        # return x.fill(1)
        return np.ones(x.shape)


class Sinus(Activation):
    def __init__(self):
        super().__init__(self.sin, self.sin_prime)

    @staticmethod
    def sin(x: Tensor):
        return np.sin(x)

    @staticmethod
    def sin_prime(s):
        return np.cos(s)


class Relu(Activation):
    def __init__(self):
        super().__init__(self.relu, self.relu_prime)

    @staticmethod
    def relu(x: Tensor):
        return np.maximum(x, 0)

    def relu_prime(self, x: Tensor):
        return np.heaviside(self.relu(x), 0)


class LeakyRelu(Activation):
    def __init__(self):
        super().__init__(self.l_relu, self.l_relu_prime)

    @staticmethod
    def l_relu(x: Tensor):
        return np.maximum(x, x * 0.01)

    @staticmethod
    def l_relu_prime(x: Tensor):
        return np.minimum(np.heaviside(x, 0) + 0.01, 1)
