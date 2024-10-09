#!/usr/bin/env python
"""
Code from chapter 19 of  Data Science from Scratch

Grus, Joel. Data Science from Scratch (p. 240). O'Reilly Media. Kindle Edition.

All code herein is either directly copied or minimally modified for my own
edification.

Usage:
    ./chapter_19.py [options]

Options:
    -h --help   Show this text.
"""

from docopt import docopt
from typing import Callable, Iterable, List, Tuple
import operator
from chapter_4 import dot
from chapter_6 import inverse_normal_cdf
from chapter_18 import sigmoid
from random import random, seed
import tqdm
from math import exp


Tensor = list  # For convenience


class Layer:
    """
    Layers of a neural network.

    Knows how to accept inputs in the "forward" direction and propagate
    gradients in the "backward" direction.

    Currently needs to be implemented by subclasses; just filler for now.
    """
    def forward(self, input):
        """
        TBD
        """
        raise NotImplementedError

    def backward(self, gradient):
        """
        TBD
        """
        raise NotImplementedError

    def params(self):
        """
        Returns the parameters of this layer. Default returns nothing.
        """
        return ()

    def grads(self):
        """
        Returns the gradients, in the same order as params().
        """
        return ()


class Sigmoid(Layer):
    """
    Sigmoid layer to use in a neural net.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply sigmoid to each element of input tensor

        Save the results to use in backpropagation.
        """
        self.sigmoids = tensor_apply(sigmoid, input)
        return self.sigmoids

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda sig, grad: sig * (1-sig) * grad,
                              self.sigmoids,
                              gradient)


class Linear(Layer):
    """Creates and initializes a linar layer for the net."""

    def __init__(self, input_dim: int, output_dim: int,
                 init: str = 'xavier') -> None:
        """
        A layer of output_dim neurons, each with input_dim weights and a bias.
        """

        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.w[o] is the weights for the oth neuron
        self.w = random_tensor(output_dim, input_dim, init=init)

        # self.b[0] is the bias term for the oth neuron
        self.b = random_tensor(output_dim, init=init)

    def forward(self, input: Tensor) -> Tensor:
        # save the input to use in the backward pass.
        self.input = input

        # return the vector of neuron outputs.
        return [dot(input, self.w[0]) + self.b[o]
                for o in range(self.output_dim)]

    def backward(self, gradient: Tensor) -> Tensor:
        # Each b[o] gets added to the output[o], which means
        # the gradient of b is the same as the output gardient.
        self.b_grad = gradient

        # Each w[o][i] multiplies input[i] and gets added to output[o].
        # So its gradient is input[i] * gradient[o].
        self.w_grad = [[self.input[i] * gradient[o]
                        for i in range(self.input_dim)]
                       for o in range(self.output_dim)]

        # Each input[i] multiplies every w[o][i] and gets added to every
        # output[o]. So its gradient is the sum of w[o][i] * gradient[o]
        # across all inputs
        return [sum(self.w[o][i] * gradient[o]
                    for o in range(self.output_dim))
                for i in range(self.input_dim)]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.b_grad]


class Sequential(Layer):
    """
    Layer consisting of a sequence of other layers.

    Will *not* confirm that the output of a layer makes sense as an input of
    the next (that's up to you).
    """
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, input):
        """Just forward the input through the layers in order."""
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, gradient):
        """Backpropagates the gradient through the layers in reverse."""
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def params(self) -> Iterable[Tensor]:
        """Just get params from each layer."""
        return (param for layer in self.layers
                for param in layer.params())

    def grads(self) -> Iterable[Tensor]:
        """Just return the grads for each layer."""
        return (grad for layer in self.layers for grad in layer.grads())


class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """How good are our predictions? (big numbers are worse.)"""
        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """How does the loss change as the predictions change?"""
        raise NotImplementedError


class SSE(Loss):
    """Loss function that computes the sum of the squared errors."""
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # Compute the tensor of wquared differences
        squared_errors = tensor_combine(
                lambda predicted, actual: 2 * (predicted - actual),
                predicted, actual)

        return tensor_sum(squared_errors)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:

        return tensor_combine(
                lambda predicted, actual: 2 * (predicted - actual),
                predicted, actual)


class Optimizer:
    """
    Updates the weights of a layer in place using information known by either
    the layer and/or the optimizer.
    """
    def step(self, layer: Layer) -> None:
        raise NotImplementedError


class GradientDescent(Optimizer):

    def __init__(self, learning_rate: float = 0.1) -> None:
        self.lr = learning_rate

    def step(self, layer:  Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            # Update param using gradient step
            param[:] = tensor_combine(
                    lambda param, grad: param - grad * self.lr,
                    param, grad)


class Momentum(Optimizer):

    def __init__(self,
                 learning_rate: float, momentum: float = 0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates = List[Tensor] = []

    def step(self, layer: Layer) -> None:
        # If we have no previous updates, start with all zeros
        if not self.updates:
            self.updates = [zeros_like(grad) for grad in layer.grads()]

        for update, param, grad in zip(self.updates,
                                       layer.params(),
                                       layer.grads()):
            # Apply momentum
            # momentum * update + (1 - momentum) * gradient
            update[:] = tensor_combine(lambda u, g: self.mo * u
                                       + (1 - self.mo) * g,
                                       update, grad)
            # Then take gradient step
            param[:] = tensor_combine(
                    lambda p, u: p - self.lr * u,
                    param, update)


class Tanh(Layer):
    def forward(self, input: Tensor) -> Tensor:
        self.tanh = tensor_apply(tanh, input)
        return self.tanh

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(
                lambda tanh, grad: (1 - tanh**2) * grad,
                self.tanh,
                gradient)


class Reul(Layer):
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return tensor_apply(lambda x: max(x, 0), input)

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda x, grad: grad if x > 0 else 0,
                              self.input
                              gradient)


def shape(tensor: Tensor) -> List[int]:
    """
    Gives you the size of each dimension of a tensor.

    Assumes that sizes and types within a tensor are consistent.
    """
    sizes: List[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes


def is_1d(tensor: Tensor) -> bool:
    """
    If tensor[0] is a list, it's a higher-order tensor.
    Otherwise, it's one-dimensional (i.e., a vector).
    """
    return not isinstance(tensor[0], list)


def tensor_sum(tensor: Tensor) -> float:
    """Sum all values in a tensor."""
    if is_1d(tensor):
        return sum(tensor)
    else:
        return sum(tensor_sum(tensor_i)  # Call tensor_sum for each row
                   for tensor_i in tensor)


def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    """Applies f elementwise"""
    if is_1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]


def zeros_like(tensor: Tensor) -> Tensor:
    """Makes a new tensor, same shape as old, but with all 0 elements."""
    return tensor_apply(lambda _: 0.0, tensor)


def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
    """Applies f to corresponding elements of t1, t2."""
    if is_1d(t1):
        return [f(x, y) for x, y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i)
                for t1_i, t2_i in zip(t1, t2)]


def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]


def random_normal(*dims: int, mean: float = 0.0,
                  variance: float = 1.0) -> Tensor:
    """Generates initial params from a random normal distribution"""
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random())
                for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance)
                for _ in range(dims[0])]


def random_tensor(*dims: int, init: str = 'normal') -> Tensor:
    if init == 'normal':
        return random_normal(*dims)
    elif init == 'uniform':
        return random_uniform(*dims)
    elif init == 'xavier':
        variance = len(dims) / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError(f"unknown init: {init}")


def train_xor():
    """
    Trains an XOR gate using gradient descent and classes in this chapter.
    """
    # training data
    # x1, x2 -> y
    # 0, 0 -> 0
    # 0, 1 -> 1
    # 1, 0 -> 1
    # 1, 1 -> 0
    xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
    ys = [[0.], [1.], [1.], [0.]]

    seed(0)

    net = Sequential([
        Linear(input_dim=2, output_dim=2),
        Sigmoid(),
        Linear(input_dim=2, output_dim=1)])

    optimizer = GradientDescent(learning_rate=0.1)
    loss = SSE()

    with tqdm.trange(3000) as t:
        for epoch in t:
            epoch_loss = 0.0

            for x, y in zip(xs, ys):
                predicted = net.forward(x)
                epoch_loss += loss.loss(predicted, y)
                gradient = loss.gradient(predicted, y)
                net.backward(gradient)

                optimizer.step(net)

            t.set_description(f"xor loss {epoch_loss: .3f}")

    for param in net.params():
        print(param)

    return net


def tanh(x: float) -> float:
    if x < -100: return -1
    elif x > 100: return 1

    em2x = exp(-2 * x)
    return (1 - em2x) / (1 + em2x)


def main() -> None:
    opts = docopt(__doc__)
    seed(0)

    # tests
    # shape()
    assert shape([1, 2, 3]) == [3]
    assert shape([[1, 2,], [1, 2], [12, 13]]) == [3, 2]

    # is_1d()
    assert is_1d([1, 2, 3])
    assert not is_1d([[1, 2], [3, 4]])

    # tensor_sum
    assert tensor_sum([1, 2, 3]) == 6
    assert tensor_sum([[1, 2], [3, 4]]) == 10

    # zeros_like()
    assert zeros_like([1, 2, 3]) == [0, 0, 0]
    assert zeros_like([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]

    # tensor_combine()
    assert tensor_combine(operator.add, [1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6]) == [4, 10, 18]

    # random_uniform()
    assert shape(random_uniform(2, 3, 4)) == [2, 3, 4]

    # random_normal()
    assert shape(random_normal(5, 6, mean=10)) == [5, 6]

    # Simple example showing assignment on a nested list
    tensor = [[1, 2], [3, 4]]

    for row in tensor:
        row = [0, 0]

    # test: nothing changes
    assert tensor == [[1, 2,], [3, 4]], "assignment doesn't update a list."
    
    for row in tensor:
        row[:] = [0, 0]

    # test: replace with zeroes
    assert tensor == [[0, 0], [0, 0]], "but slice assignment does."

    train_xor()


if __name__ == '__main__':
    main()
