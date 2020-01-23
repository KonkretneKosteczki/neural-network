from numpy import ndarray as Tensor
from lab3.neuralnet import NeuralNet
from lab3.loss import Loss, TotalSquaredError
from lab3.optimizer import Optimizer, GradientDescent
from lab3.data import DataIterator, BatchIterator


def train(net: NeuralNet, inputs: Tensor, targets: Tensor, iterations: int = 50,
          iterator: DataIterator = BatchIterator(), loss: Loss = TotalSquaredError(),
          optimizer: Optimizer = GradientDescent()) -> None:
    for iteration in range(iterations):
        iter_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            iter_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net, iteration)
        # print(epoch, epoch_loss)
