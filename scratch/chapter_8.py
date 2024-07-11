#!/usr/bin/env python
"""
Code from Chatper 8 of Data Science from Scratch.

Usage:
    ./chapter_8.py [options]

Options:
    -h --help   Show this text.
"""

import random
from docopt import docopt
from chapter_4 import add, distance, dot, scalar_multiply, Vector, vector_mean
from typing import Callable
from math import isnan

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x+h) - f(x)) / h


def partial_difference_quotient(f: Callable[[Vector], float], v: Vector,
                                i: int, h: float) -> float:
    """
    Partial derivative of 'f' w.r.t. the 'i' component of the vector.

    Step size is 'h'.
    """
    w = [v_j + (dq if j == i else 0)
         for j, v_j in enumerate(v)]
    return (f(w) - v(v)) / h


def estimate_gradient(f: Callable[[Vector], float], v: Vector,
                      h: float = 0.0001):
    """
    Estimates the gradient of the function f at point v numerically.

    Step size is h.
    """
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


def sum_of_squares(v: Vector) -> float:
    """Computes the magnitude squared of a vector."""
    return dot(v, v)


def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves the vector a step along the gradient"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    """
    Very specific function used to solve a simple problem in chapter 8.

    Determines gradient using a "true" data point.

    The vector theta is (slope, intercept). X and y are data points we use to
    compute our error squared.
    """
    slope, intercept = theta
    predicted = slope * x + intercept  # What the model predicts
    error = (predicted - y)  # difference between prediction & actual value
    squared_error = error * error  # we'll minimize this
    # f = (slope * x + intercept - y)^2
    #   = (slope^2 * x^2 + 2*slope*x*intercept - 2*slope*x*y + intercept^2
    #      - 2 * intercept * y + y^2)
    # df/d(slope) = 2 * slope * x^2 + 2*x*intercept - 2*x*y
    #             = 2x(slope*x + intercept - y)
    #             = 2*x*(predicted - y)
    #             = 2*x*error
    # df/d(intercept) = 2*slope*x + 2*intercept - 2*y
    #                 = 2*(predicted - y)
    #                 = 2*(error)
    grad = [2 * error * x, 2 * error]  
    return grad


def main() -> None:
    opts = docopt(__doc__)

    # First scenario: finding the minimum of f = dot(v, v)
    # random starting point for a three-dimensional vector
    v = [random.uniform(-10, 10) for i in range(3)]
    print(f"Finding the minimum of a vector field. Starting at the point "
          f"{v}.")

    for epoch in range(1000):
        grad = sum_of_squares_gradient(v)
        v = gradient_step(v, grad, -0.01)  # Finding a nadir
        print(epoch, v)

    assert distance(v, [0, 0, 0]) < 0.001  # should move close to the origin

    # second scenario: finding slope/intercept using a guess

    # Actual data: y = 20 * x + 5
    # So... slope is 20, intercept is 5.
    # x goes from -50 to 49, y(x) = 20 * x + 5
    inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

    # Begin with a random guess for slope/intercept
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    slope, intercept = theta

    learning_rate = 0.001

    print(f"Finding the slope & intercept of a field. Starting at "
          f"slope = {slope}, intercept = {intercept}")
    # This finds the slope quickly but takes awhile to find the intercept.
    for epoch in range(1000):

        # Mean gradient of all points x, y in `inputs`
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])

        # Take a step *opposite* the gradient
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
        if isnan(theta[0]) or isnan(theta[1]): break

    slope, intercept = theta
    assert 19.9 < slope < 20.1, f"Expected slope was 20, actual was {slope}"
    assert 4.9 < intercept < 5.1, f"Expected intercept was 5, actual was {intercept}"


if __name__ == '__main__':
    main()
