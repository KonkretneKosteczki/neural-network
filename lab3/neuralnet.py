# just a collection of layers
from numpy import ndarray as Tensor
from lab3.neuron import Layer
from typing import Sequence, Iterator, Tuple
from lab3.loss import Loss, TotalSquaredError
from lab3.data import DataIterator, BatchIterator


class NeuralNet:
    def __init__(self, layers: Sequence[Layer], lr: float = 0.01) -> None:
        self.layers = layers
        self.lr = lr

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

    def gradient_descent(self, iteration: int) -> None:
        for param, grad in self.params_and_grads():
            param -= self.lr / (iteration + 1) * grad  # variable learning rate

    def train(self, inputs: Tensor, targets: Tensor, iterations: int = 50,
              iterator: DataIterator = BatchIterator(), loss: Loss = TotalSquaredError()) -> None:
        for iteration in range(iterations):
            iter_loss = 0.0
            for batch in iterator(inputs, targets):
                predicted = self.forward(batch.inputs)
                iter_loss += loss.loss(predicted, batch.targets)
                grad = loss.grad(predicted, batch.targets)
                self.backward(grad)
                self.gradient_descent(iteration)
