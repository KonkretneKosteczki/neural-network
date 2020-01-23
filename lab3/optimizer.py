# update parameters of network based on gradient computed during backpropagation
from lab3.neuralnet import NeuralNet


class Optimizer:
    def step(self, net: NeuralNet, iteration: int) -> None:
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, net: NeuralNet, iteration: int) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr / (iteration + 1) * grad  # variable learning rate
