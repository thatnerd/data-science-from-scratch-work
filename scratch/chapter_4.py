#!/usr/bin/env python
"""
Code I input in chapter 4 of Data Science from Scratch

Usage:
    ./chapter_4.py [options]

Options:
    -h --help   Show this text.
"""

from docopt import docopt
from typing import Callable, List, Tuple
import math


Vector = List[float]
Matrix = List[Vector]


def add(v: Vector, w: Vector) -> Vector:
    """Adds the corresponding elements of each vector"""
    assert len(v) == len(w), "Vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements of w from v"""
    assert len(v) == len(w)
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements of the vectors"""
    assert vectors, "No vectors provided"

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "vectors passed have different sizes!"

    return [sum(vector[i] for vector in vectors) 
            for i in range(num_elements)]


def scalar_multiply(c: float, v: Vector) -> Vector:
    """Scales the vector by a constant"""
    return [c * element for element in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the 'average' element from a bunch of vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


def dot(v: Vector, w: Vector) -> float:
    """Computes the dot product of two vectors"""
    assert len(v) == len(w), "vectors must be the same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v: Vector) -> float:
    """Returns the magnitude squared of a vector"""
    return dot(v, v)


def magnitude(v: Vector) -> float:
    """Returns the magnitude of a vector"""
    return math.sqrt(sum_of_squares(v))


def squared_distance(v: Vector, w: Vector) -> float:
    """Finds the square of the distance between two vectors"""
    return sum_of_squares(subract(v, w))



def distance(v: Vector, w: Vector) -> float:
    """Returns the distance between two vectors"""
    return magnitude(subtract(v, w))


def shape(A: Matrix) -> Tuple[int, int]:
    """Returns the number of rows & columns of a matrix"""
    return len(A), len(A[0]) if A else 0


def get_row(A: Matrix, i: int) -> Vector:
    """Grabs one row of a matrix"""
    return A[i]


def get_column(A: Matrix, i: int) -> Vector:
    """Grabs one column of a matrix"""
    return [A_i[j] for A_i in A]


def make_matrix(num_rows: int, num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """Returns a matrix defined by entry_fn(i, j)"""
    return [[entry_fn(i, j)
             for j in range(num_cols)]
            for i in range(num_rows)]


def identity_matrix(n: int) -> Matrix:
    """Creates a square identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i==j else 0)


def main() -> None:
    opts = docopt(__doc__)

    # We don't even use these
    height_weight_age = [70, 170, 40]  # inches, pounds, years
    grades = [95, 80, 75, 62]

    # author's putting in unit tests for vectors; putting them here
    assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]
    assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]
    assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]
    assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]
    assert dot([1, 2, 3], [4, 5, 6]) == 32
    assert sum_of_squares([1, 2, 3]) == 14
    assert magnitude([3, 4]) == 5
    assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
    assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1]]

    A = [[1, 2, 3],
         [4, 5, 6]]

    B = [[1, 2],
         [3, 4],
         [5, 6]]




if __name__ == '__main__':
    main()
