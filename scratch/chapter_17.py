#!/usr/bin/env python
"""
Run for the scripts of chapter 17 of DS from S

Usage:
    ./chapter_17.py [options]

Options:
    -h --help   Show this text.
"""

from docopt import docopt
from math import log
from typing import List, Any, NamedTuple, Optional, Dict, TypeVar, Union
from collections import Counter, defaultdict


def entropy(class_probabilities: List[float]) -> float:
    """Given a list of probabilities, compute the entropy"""
    assert sum(class_probabilities) == 1.0
    return sum(-p * log(p, 2) for p in class_probabilities
               if 0 < p < 1)  # just ignore bad probabilities


def class_probabilities(labels: List[Any]) -> List[float]:
    """
    Finds the probabilities of selecting each label in a list of labels.
    """
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]


def data_entropy(labels: List[Any]) -> float:
    """Finds the entropy of a list of labels."""
    return entropy(class_probabilities(labels))


def partition_entropy(subsets: List[List[Any]]) -> float:
    """Returns the entropy from this partition of data into subsets."""
    total_count = sum(len(subset) for subset in subsets)

    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)


class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None  # allow unlabeled data


T = TypeVar('T')  # generic type for inputs


def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List]:
    """Partition the inputs into lists based on the specified attribute."""
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)  # value of the specified attribute
        partitions[key].append(input)  # add input to the correct partition
    return partitions


def partition_entropy_by(inputs: List[Any],
                         attribute: str,
                         label_attribute: str) -> float:
    """Compute entropy corresponding to the given partition"""
    # partitions consist of our inputs
    partitions = partition_by(inputs, attribute)

    # but partition_entropy just needs the class labels
    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]

    return partition_entropy(labels)


class Leaf(NamedTuple):
    value: Any

class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None

DecisionTree = Union[Leaf, Split]

hiring_tree = Split('level', {
    'Junior': Split('phd', {
        False: Leaf(True),
        True: Leaf(False)
    }),
    'Mid': Leaf(True),
    'Senior': Split('tweets', {
        False: Leaf(False),
        True: Leaf(True)
    })
})


def classify(tree: DecisionTree, input: Any) -> Any:
    """Classify the input using the given decision tree"""

    if isinstance(tree, Leaf):
        return tree.value

    # otherwise, tree consists of an attribute to split on
    # and a dict whose keys are values of that attribute
    # and whose values are the subtrees to consider next
    subtree_key = getattr(input, tree.attribute)

    if subtree_key not in tree.subtrees:  # If no subtree for key
        return tree.default_value  # return default

    subtree = tree.subtrees[subtree_key]  # Choose the right subtree
    return classify(subtree, input)  # Classify the input

def build_tree_id3(inputs: List[Any],
                   split_attributes: List[str],
                   target_attribute: str) -> DecisionTree:
    # Count target labels
    label_counts = Counter(getattr(input, target_attribute)
                           for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]

    # Predict if there's a unique label
    if len(label_counts) == 1:
        return Leaf(most_common_label)

    # If no split attributes are left, return the majority label
    if not split_attributes:
        return Leaf(most_common_label)

    # Otherwise: split by the best attribute

    def split_entropy(attribute: str) -> float:
        """Helper function for finding the best attribute"""
        return partition_entropy_by(inputs, attribute, target_attribute)

    best_attribute = min(split_attributes, key=split_entropy)

    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]

    # Recursively build the subtrees
    subtrees = {attribute_value: build_tree_id3(subset,
                                               new_attributes,
                                               target_attribute)
                for attribute_value, subset in partitions.items()}

    return Split(best_attribute, subtrees, default_value=most_common_label)


def main() -> None:
    opts = docopt(__doc__)
    assert entropy([1.0]) == 0
    assert entropy([0.5, 0.5]) == 1
    assert 0.81 < entropy([0.25, 0.75]) < 0.82

    # copied from text
    inputs = [Candidate('Senior', 'Java',   False, False, False),
          Candidate('Senior', 'Java',   False, True,  False),
          Candidate('Mid',    'Python', False, False, True),
          Candidate('Junior', 'Python', False, False, True),
          Candidate('Junior', 'R',      True,  False, True),
          Candidate('Junior', 'R',      True,  True,  False),
          Candidate('Mid',    'R',      True,  True,  True),
          Candidate('Senior', 'Python', False, False, False),
          Candidate('Senior', 'R',      True,  False, True),
          Candidate('Junior', 'Python', True,  False, True),
          Candidate('Senior', 'Python', True,  True,  True),
          Candidate('Mid',    'Python', False, True,  True),
          Candidate('Mid',    'Java',   True,  False, True),
          Candidate('Junior', 'Python', False, True,  False)]

    for key in ['level', 'lang', 'tweets', 'phd']: 
        print(key, partition_entropy_by(inputs, key, 'did_well'))

    assert 0.69 < partition_entropy_by(inputs, 'level', 'did_well') < 0.70
    assert 0.86 < partition_entropy_by(inputs, 'lang', 'did_well') < 0.87
    assert 0.78 < partition_entropy_by(inputs, 'tweets', 'did_well') < 0.79
    assert 0.89 < partition_entropy_by(inputs, 'phd', 'did_well') < 0.90

    senior_inputs = [input for input in inputs if input.level == 'Senior']

    assert 0.4 == partition_entropy_by(senior_inputs, 'lang', 'did_well')
    assert 0.0 == partition_entropy_by(senior_inputs, 'tweets', 'did_well')
    assert 0.95 < partition_entropy_by(senior_inputs, 'phd', 'did_well') < 0.96

    tree = build_tree_id3(inputs,
                          ['level', 'lang', 'tweets', 'phd'],
                          'did_well')

    assert classify(tree, Candidate('Junior', 'Java', True, False))



if __name__ == '__main__':
    main()
