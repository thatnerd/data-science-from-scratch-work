#!/usr/bin/env python
"""
Code from Chapter 12.

Usage:
    ./chapter_12.py [options]

Options:
    -h --help   Show this text.
"""

from docopt import docopt
from typing import Dict, List, NamedTuple, Tuple
from collections import Counter
from chapter_4 import Vector, distance
import requests
import csv
from collections import defaultdict
from matplotlib import pyplot as plt
import random
from chapter_11 import split_data
import tqdm


data = requests.get(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)


class LabeledPoint(NamedTuple):
    point: Vector
    label: str


def knn_classify(k: int,
                 labeled_points: List[LabeledPoint],
                 new_point: Vector) -> str:
    """Classify points into nearest neighbors"""
    by_distance = sorted(labeled_points,
                         key=lambda lp: distance(lp.point, new_point))

    # Find the labels for the closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    # let them vote
    return majority_vote(k_nearest_labels)


def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner


def majority_vote(labels: List[str]) -> str:
    """Assumes that labels are ordered from nearest to furthest."""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])
    if num_winners == 1:
        return winner  # unique winner, so return it
    else:
        # Could make this function call expensive!
        return majority_vote(labels[:-1])  # try again without the farthest


def parse_iris_row(row: List[str]) -> LabeledPoint:
    """
    sepal_length, sepal_width, petal_length, petal_width, class
    """
    measurements = [float(value) for value in row[:-1]]
    # class is e.g. "Iris-virginica"; we just want "Virginica"
    label = row[-1].split("-")[-1]

    return LabeledPoint(measurements, label)


def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]


def random_distances(dim: int, num_pairs: int) -> List[float]:
    return [distance(random_point(dim), random_point(dim))
            for _ in range(num_pairs)]


def main() -> None:
    opts = docopt(__doc__)
    assert raw_majority_vote(['a', 'b', 'c', 'b']) == 'b'
    # Just needed to run this one time
    # with open('iris.dat', 'w') as f:
    #     f.write(data.text)

    with open('iris.dat') as f:
        reader = csv.reader(f)
        iris_data = [parse_iris_row(row) for row in reader]

    # Group just the points by species/label so we can plot them
    points_by_species: Dict[str, List[Vector]] = defaultdict(list)
    for iris in iris_data:
        points_by_species[iris.label].append(iris.point)

    # # Set up and plot the irises
    # metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']
    # pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]
    # marks = ['+', '.', 'x']  # 3 classes, so 3 markers
    # fig, ax = plt.subplots(2, 3)

    # for row in range(2):
    #     for col in range(3):
    #         i, j = pairs[3 * row + col]
    #         ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)
    #         ax[row][col].set_xticks([])
    #         ax[row][col].set_yticks([])
    #         for mark, (species, points) \
    #                 in zip(marks, points_by_species.items()):
    #             xs = [point[i] for point in points]
    #             ys = [point[j] for point in points]
    #             ax[row][col].scatter(xs, ys, marker=mark, label=species)

    # ax[-1][-1].legend(loc='lower right', prop={'size': 6})
    # plt.show()

    random.seed(12)
    iris_train, iris_test = split_data(iris_data, 0.7)
    assert len(iris_train) == 0.7 * 150
    assert len(iris_test) == 0.3 * 150

    # track how many times we see (predicted, actual)
    confusion_matrix: Dict[tuple[str, str], int] = defaultdict(int)
    num_correct = 0

    for iris in iris_test:
        predicted = knn_classify(5, iris_train, iris.point)
        actual = iris.label

        if predicted == actual:
            num_correct += 1

        confusion_matrix[(predicted, actual)] += 1

    pct_correct = num_correct / len(iris_test)
    print(pct_correct, confusion_matrix)
    dimensions = range(1, 101)

    avg_distances = []
    min_distances = []

    random.seed(0)
    for dim in tqdm.tqdm(dimensions, desc="Curse of Dimensionality"):
        distances = random_distances(dim, 10000)  # 10k random pairs
        avg_distances.append(sum(distances) / 10000)  # track avg
        min_distances.append(min(distances))   # track the min

    min_avg_ratio = [min_dist / avg_dist
                     for min_dist, avg_dist in zip(min_distances,
                                                   avg_distances)]
    

    



if __name__ == '__main__':
    main()
