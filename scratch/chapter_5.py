#!/usr/bin/env python
"""
Scratch work from chapter 5

Usage:
    ./chapter_5.py [options]

Options:
    -h --help   Show this text.
"""

from docopt import docopt
from chapter_4 import dot, sum_of_squares
from typing import List
import math


# copy/pasted from author's repo, as opposed to most of this module, which is
# copied by hand with maybe some improv
num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
daily_minutes = [1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]
daily_hours = [dm / 60 for dm in daily_minutes]


def mean(xs: List[float]) -> float:
    """Finds the statistical mean of a list of numbers."""
    return sum(xs) / len(xs)


def de_mean(xs: List[float]) -> List[float]:
    """Shift the xs so that the mean is now zero."""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def variance(xs: list[float]) -> float:
    """Almost the average squared devation from the mean."""
    assert len(xs) >=2, "Variance requires at least two elements."
    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)


def standard_deviation(xs: List[float]) -> float:
    """Implementing this manually."""
    return math.sqrt(variance(xs))


def covariance(xs: List[float], ys: List[float]) -> float:
    """Manual implementation of what should be called in a library."""
    assert len(xs) == len(ys)
    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)


def correlation(xs: List[float], ys: List[float]) -> float:
    """Manual implementation of what should be in a library."""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0  # If no variation, no correlation.


def main() -> None:
    opts = docopt(__doc__)
    assert 81.54 < variance(num_friends) < 81.55
    print(f"Variance of num_friends is: {variance(num_friends)}")
    assert 9.02 < standard_deviation(num_friends) < 9.04
    print(f"Standard deviation of num_friends is: {standard_deviation(num_friends)}")
    print(f"Covariance of num_friends with daily_minutes is: {covariance(num_friends, daily_minutes)}")
    assert 22.42 < covariance(num_friends, daily_minutes) < 22.43
    assert 22.42 / 60 < covariance(num_friends, daily_hours) < 22.43 / 60
    assert 0.24 < correlation(num_friends, daily_minutes) < 0.25
    assert 0.24 < correlation(num_friends, daily_hours) < 0.25
    print(f"Pearson's R of num_friends with daily_minutes is: {correlation(num_friends, daily_minutes)}")
    outlier = num_friends.index(100)  # It's an outlier and 100 is its index.
    num_friends_good = [x for i, x in enumerate(num_friends)
                        if i != outlier]
    daily_minutes_good = [x for i, x in enumerate(daily_minutes)
                          if i != outlier]
    daily_hours_good = [dm/60 for dm in daily_minutes_good]
    print(f"Pearson's R of num_friends with daily_minutes without an outlier is: {correlation(num_friends_good, daily_minutes_good)}")
    assert 0.57 < correlation(num_friends_good, daily_minutes_good) < 0.58
    assert 0.57 < correlation(num_friends_good, daily_hours_good) < 0.58



if __name__ == '__main__':
    main()
