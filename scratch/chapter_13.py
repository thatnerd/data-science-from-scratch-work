#!/usr/bin/env python
"""
Chapter 13 python work from Data Science from Scratch.

Grus, Joel. Data Science from Scratch. O'Reilly Media. Kindle Edition.

Usage:
    ./chapter_13.py [options]

Options:
    -h --help   Show this text.
"""

from docopt import docopt
from typing import List, Tuple, Dict, Iterable, Set, NamedTuple
from collections import Counter, defaultdict
import math
import re
from io import BytesIO
import requests
import tarfile
import glob
import random
from chapter_11 import split_data

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = ["20021010_easy_ham.tar.bz2",
         "20021010_hard_ham.tar.bz2",
         "20021010_spam.tar.bz2"]

OUTPUT_DIR = "spam_data"


class Message(NamedTuple):
    text: str
    is_spam: bool


class NaiveBayesClassifier:

    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor
        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # increment message counts
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """Returns P(token | spam) and P(token | ham)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0

        # Iterate through each word in our vocab
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # If *token* appears in the message,
            # add the log probability of seeing it
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
            else:  # Add log(probability of *not* seeing it)
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)


def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9']+", text)  # extract all words
    return set(all_words)  # remove dupes


def download_and_unpack_spam_assassin_public_corpus() -> None:
    """Does what it says on the tin."""
    for filename in FILES:
        content = requests.get(f"{BASE_URL}/{filename}").content

        fin = BytesIO(content)

        with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
            tf.extractall(OUTPUT_DIR)

def get_subject_lines(path: str = 'spam_data/*/*') -> List[Message]:
    data: List[Message] = []

    for filename in glob.glob(path):
        is_spam = "ham" not in filename

        with open(filename, errors="ignore") as email_file:
            for line in email_file:
                if line.startswith("Subject:"):
                    subject = line.lstrip("Subject: ")
                    data.append(Message(subject, is_spam))
                    break  # found the subject

    return data


def train_spam_filter_on_messages(train_messages: List[Message]):
    model = NaiveBayesClassifier()
    model.train(train_messages)
    return model


def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    prob_if_spam, prob_if_ham = model._probabilities(token)

    return prob_if_spam / (prob_if_spam + prob_if_ham)


def main() -> None:
    opts = docopt(__doc__)

    assert tokenize("Data Science is science") == {"data", "science", "is"}

    messages = [Message("spam rules", is_spam=True),
                Message("ham rules", is_spam=False),
                Message("Hello ham", is_spam=False)]

    model = NaiveBayesClassifier(k=0.5)
    model.train(messages)

    assert model.tokens == {"spam", "ham", "rules", "hello"}
    assert model.spam_messages == 1
    assert model.ham_messages == 2
    assert model.token_spam_counts == {"spam": 1, "rules": 1}
    assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

    text = "hello spam"

    probs_if_spam = [(1 + 0.5) / (1 + 2 * 0.5),  # spam (present)
                     1 - (0 + 0.5) / (1 + 2 * 0.5),  # ham (not present)
                     1 - (1 + 0.5) / (1 + 2 * 0.5),  # rules (not present)
                     (0 + 0.5) / (1 + 2 * 0.5)]  # hello

    probs_if_ham = [(0 + 0.5) / (2 + 2 * 0.5),  # spam (present)
                    1 - (2 + 0.5) / (2 + 2 * 0.5),  # ham (not present)
                    1 - (1 + 0.5) / (2 + 2 * 0.5),  # rules (not present)
                    (1 + 0.5) / (2 + 2 * 0.5)]  # Hello (present)


    p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
    p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

    assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham)

    # Download the data just once and it'll be in place
    # download_and_unpack_spam_assassin_public_corpus()

    data: List[Message] = get_subject_lines()

    random.seed(0)
    train_messages, test_messages = split_data(data, 0.75)

    model = train_spam_filter_on_messages(train_messages)
    predictions = [(message, model.predict(message.text))
                   for message in test_messages]

    confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                               for message, spam_probability in predictions)

    print("This is not currently giving me the numbers in the book:")
    print(confusion_matrix)

    words = sorted(model.tokens, key = lambda t: p_spam_given_token(t, model))

    print("spammiest_words", words[-10:])
    print("hammiest_words", words[:10])



if __name__ == '__main__':
    main()
