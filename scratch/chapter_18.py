#!/usr/bin/env python
"""
Code from Chapter 18, DS from S.

Grus, Joel. Data Science from Scratch (p. 227-). O'Reilly Media. Kindle Edition. 

Code either copied by hand or modified somewhat for my own coding style.

Usage:
    ./chapter_18.py [options]

Options:
    -h --help   Show this text.
"""

from docopt import docopt
from chapter_4 import Vector, dot, squared_distance
from math import exp
from typing import List
import random
from chapter_8 import gradient_step
import tqdm
from pprint import pprint


def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0.0


def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
    """Determines if the perceptron 'fires' or not."""
    calculation = dot(weights, x) + bias
    return step_function(calculation)


def simulate_and_gate(input_1: int, input_2: int) -> int:
    weights = [2., 2]
    bias = -3.
    return perceptron_output(weights, bias, [input_1, input_2])


def simulate_or_gate(input_1: int, input_2: int) -> int:
    weights = [2., 2]
    bias = -1
    return perceptron_output(weights, bias, [input_1, input_2])


def simulate_not_gate(input_1: int) -> int:
    weights = [-2.]
    bias = 1.
    return perceptron_output(weights, bias, [input_1])


def sigmoid(t: float) -> float:
    return 1 / (1 + exp(-t))


def neuron_output(weights: Vector, inputs: Vector) -> float:
    # weights includes the bias term, inputs includes a 1
    return sigmoid(dot(weights, inputs))


def feed_forward(neural_network: List[List[Vector]],
                 input_vector: Vector) -> List[Vector]:
    """
    Feeds the input through the network.
    
    Returns all layers (not just the final one)
    """
    outputs: List[Vector] = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]  # Add a constant
        output = [neuron_output(neuron, input_with_bias)  # Compute the output
                  for neuron in layer]
        outputs.append(output)

        # Input to the next layer is the output of this one
        input_vector = output

    return outputs


def simulate_xor_gate(input_1: int, input_2: int) -> int:
    """
    Uses a pre-defined neural network to get the XOR behavior.

    Weightings are arbitrary and used to create sigmoid values close to 1 and
    0.
    """
    # 'and' neuron:
    # Truth table:
    #   1, 1 -> 1 (20 + 20 - 30 =  10)
    #   1, 0 -> 0 (20 + 00 - 30 = -10)
    #   0, 1 -> 0 (00 + 20 - 30 = -10)
    #   0, 0 -> 0 (00 + 00 - 30 = -30)
    # both inputs must be 1 to overcome the bias
    and_neuron = [20., 20, -30]  

    # 'or' neuron next:
    # either input will overcome the bias if nonzero
    # Truth table:
    #   1, 1 -> 1  (20 + 20 - 10 =  30)
    #   1, 0 -> 1  (20 + 00 - 10 =  10)
    #   0, 1 -> 1  (00 + 20 - 10 =  10)
    #   0, 0 -> 0  (00 + 00 - 10 = -10)
    or_neuron = [20., 20, -10]

    # 'resolution' neuron next
    # True only if the 2nd neuron is true and first neuron is false.
    # Use 'and' and 'or' as its 1st and 2nd inputs, respectively.
    # Truth table:
    #   1, 1 -> 0  (-60 + 60 - 30 = -30)
    #   1, 0 -> 0  (-60 + 00 - 30 = -90)
    #   0, 1 -> 1  ( 00 + 60 - 30 =  30)
    #   0, 0 -> 0  ( 00 + 00 - 30 = -30)
    resolution_neuron = [-60., 60, -30]

    # Three-layer network:
    # 1. Inputs (two boolean inputs)
    # 2. AND and OR nodes
    # 3. Resolution node: 2 & !1
    xor_network = [
        # first is the hidden layer: two neurons that calculate
        # 'and' and 'or'
        [and_neuron, or_neuron], 
        [resolution_neuron]  # '2nd input and not 1st input' neuron
    ]  
    # Return just the last layer but first node (it's the only one we created)
    return feed_forward(xor_network, [input_1, input_2])[-1][0]


def sqerror_gradients(network: List[List[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:
    """
    Makes a prediction and checks to see its error squared.
    """
    # forward pass
    # Initial confusion: why is feed_forward giving hidden outputs *and*
    #   outputs?
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # Gradients w.r.t. output neuron pre-activation outputs
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]

    # Gradients w.r.t. output neuron weights
    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]

    # Gradients w.r.t. hidden neuron pre-activation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) * 
                     dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # Gradients w.r.t. hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                     for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]


def learn_xor_network(seed: int = 0) -> List[List[float]]:
    random.seed(seed)

    # I/O logic we want: xs -> ys
    xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
    ys = [[0.], [1.], [1.], [0.]]

    # Start with random weights
    network = [
        # hidden neurons
        # 2 + 1 because 2 inputs, +1 for the algorithm
        [[random.random() for _ in range(2 + 1)],  # 1st hidden neuron
         [random.random() for _ in range(2 + 1)]],  # 2nd hidden neuron
        # output layer: 2 inputs -> 1 output
        [[random.random() for _ in range(2 + 1)]]  # 1st output neuron
    ]

    learning_rate = 1.0

    for epoch in tqdm.trange(20000, desc="neural net for xor"):
        for x, y in zip(xs, ys):
            gradients = sqerror_gradients(network, x, y)

            # Take a step for each neuron in each layer
            network = [[gradient_step(neuron, grad, -learning_rate)
                        for neuron, grad in zip(layer, layer_grad)]
                       for layer, layer_grad in zip(network, gradients)]

    return network


def fizz_buzz_labels(x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def generate_binary_digits(x: int) -> Vector:
    """
    Finds the bigendian binarry digits of x, up to 2^9
    """
    # First, find the 10-digit binary encoding of the input
    # zfill adds leading 0's to make a string a certain length
    binary_string: str = bin(x)[2:].zfill(10)
    # turn the encoding into a string
    binary_digits: List[float] = [int(c) for c in binary_string] 
    # Make it bigendian
    binary_digits.reverse()
    return binary_digits


def learn_fizz_buzz_network(seed: int = 0,
                            hidden:int = 25) -> List[List[float]]:
    xs = [generate_binary_digits(n) for n in range(101, 1024)]
    ys = [fizz_buzz_labels(n) for n in range(101, 1024)]

    # Generate 25-node neural network with random weights
    random.seed(seed)
    learning_rate = 1.0
    network = [
        # hidden layer: 10 inputes -> "hidden" outputs (i.e., 25)
        [[random.random() for _ in range(10 + 1)] for _ in range(hidden)],

        # output: "hidden" inputs (i.e., 25) -> 4 outputs
        [[random.random() for _ in range(hidden + 1)] for _ in range(4)]
    ]

    with tqdm.trange(500) as t:
        for epoch in t:
            epoch_loss = 0.0

            for x, y in zip(xs, ys):
                predicted = feed_forward(network, x)[-1]
                epoch_loss += squared_distance(predicted, y)
                gradients = sqerror_gradients(network, x, y)

                # Gradient step for each neuron in each layer
                network = [[gradient_step(neuron, grad, -learning_rate)
                            for neuron, grad in zip(layer, layer_grad)]
                           for layer, layer_grad in zip(network, gradients)]

        t.set_description(f"fizz buzz (loss: {epoch_loss: .2f})")
    
    return network


def argmax(xs: list) -> int:
    """Used to decode the fizz buzz answers."""
    return max(range(len(xs)), key = lambda i: xs[i])



def main() -> None:
    opts = docopt(__doc__)

    # test the "AND Gate" simulator
    assert simulate_and_gate(1, 1) == 1
    assert simulate_and_gate(0, 1) == 0
    assert simulate_and_gate(1, 0) == 0
    assert simulate_and_gate(0, 0) == 0

    # test the "OR Gate" simulator
    assert simulate_or_gate(1, 1) == 1
    assert simulate_or_gate(0, 1) == 1
    assert simulate_or_gate(1, 0) == 1
    assert simulate_or_gate(0, 0) == 0

    # test the "NOT Gate" simulator
    assert simulate_not_gate(1) == 0
    assert simulate_not_gate(0) == 1

    # Test the "XOR Gate" neural net simulator
    assert 0.00 < simulate_xor_gate(1, 1) < 0.001
    assert 0.999 < simulate_xor_gate(1, 0) < 1.0
    assert 0.999 < simulate_xor_gate(0, 1) < 1.0
    assert 0.00 < simulate_xor_gate(0, 0) < 0.001

    # Use gradient descent to train a network for XOR
    network = learn_xor_network()

    # Check that it learned XOR
    assert feed_forward(network, [0, 0])[-1][0] < 0.01
    assert feed_forward(network, [0, 1])[-1][0] > 1 - 0.01
    assert feed_forward(network, [1, 0])[-1][0] > 1 - 0.01
    assert feed_forward(network, [1, 1])[-1][0] < 0.01

    print("This is what the neural network built for XOR looks like:")
    pprint(network)

    # Use gradient descent to train a network for fizz buzz
    # Copied the following test from the textbook
    #                             1  2  4  8 16 32 64 128 256 512
    assert generate_binary_digits(0)   == [0, 0, 0, 0, 0, 0, 0, 0,  0,  0]
    assert generate_binary_digits(1)   == [1, 0, 0, 0, 0, 0, 0, 0,  0,  0]
    assert generate_binary_digits(10)  == [0, 1, 0, 1, 0, 0, 0, 0,  0,  0]
    assert generate_binary_digits(101) == [1, 0, 1, 0, 0, 1, 1, 0,  0,  0]
    assert generate_binary_digits(999) == [1, 1, 1, 0, 0, 1, 1, 1,  1,  1]
    # Source: Grus, Joel. Data Science from Scratch (p. 236). O'Reilly Media. Kindle Edition.

    network = learn_fizz_buzz_network()

    print("This is what the neural network built for Fizz Buzz looks like:")
    print("Not showing network because it's hard to visually parse")
    # pprint(network)

    # Just using this for argmax; same as in learn_fizz_buzz_network()
    xs = [generate_binary_digits(n) for n in range(101, 1024)]

    assert argmax([0, -1]) == 0  # Item 0 is largest
    assert argmax([-1, 0]) == 1  # Item 1 is largest
    assert argmax([-1, 10, 5, 20, -3]) == 3  # item 3 is largest

    num_correct = 0

    for n in range(1, 101):
        x = generate_binary_digits(n)
        predicted = argmax(feed_forward(network, x)[-1])
        actual = argmax(fizz_buzz_labels(n))
        labels = [str(n), "fizz", "buzz", "fizzbuzz"]
        print(n, labels[predicted], labels[actual])
        if predicted == actual:
            num_correct += 1

    print(num_correct, "/", 100)


if __name__ == '__main__':
    main()
