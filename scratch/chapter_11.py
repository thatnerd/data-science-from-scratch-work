#!/usr/bin/env python
"""
Work from Chapter 11 of Grus, Joel. Data Science from Scratch (p. 156). O'Reilly Media. Kindle Edition.

Hand-copied from the book; data examples may be copy/pasted from public repo: https://github.com/joelgrus/data-science-from-scratch

This is definitely not my work, but I like to type in everything by hand I'm
copying, since I regularly learn from the process and have to debug my typos at
minimum.

Usage:
    ./chapter_11.py [options]

Options:
    -h --help   Show this text.
"""

from docopt import docopt
import random
from typing import TypeVar, List, Tuple


X = TypeVar('X')  # Generic type, represents a single data point
Y = TypeVar('Y')  # Generic type, represents output variables


def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """
    Shuffles & splits data into two bins
    """
    # shallow copy data to a new list to not alter input in place
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)  # index to split training data & test data
    return data[:cut], data[cut:]


def train_test_split(xs: List[X], ys: List[Y], test_pct: float
                     ) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    """Split pairs of data, as with split_data()."""
    idxs = [i for i in range(len(xs))]

    # split the indices, not the data
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],
            [xs[i] for i in test_idxs],
            [ys[i] for i in train_idxs],
            [ys[i] for i in test_idxs])


def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    """
    Accuracy of a test in terms of:
    * true positive
    * false positive
    * false negative
    * true negative

    This is probably *way* too general of a function name for this type of
    function.
    """
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)


def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)


def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    """Harmonic mean of precision and recall """


def main() -> None:
    opts = docopt(__doc__)

    data = [n for n in range(1000)]
    train, test = split_data(data, 0.75)

    assert len(train) == 750
    assert len(test) == 250
    # Confirm OG data is preserved
    assert sorted(train + test) == data

    xs = [x for x in range(1000)]
    ys = [2 * x for x in xs]
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_pct=0.25)

    assert len(x_train) == len(y_train) == 750
    assert len(x_test) == len(y_test) == 250

    assert all(y == 2 * x for x, y in zip(x_train, y_train))
    assert all(y == 2 * x for x, y in zip(x_test, y_test))

    assert accuracy(70, 4930, 13930, 981070) == 0.98114

    assert precision(70, 4930, 13930, 981070) == 0.014

    assert recall(70, 4930, 13930, 981070) == 0.005



if __name__ == '__main__':
    main()
