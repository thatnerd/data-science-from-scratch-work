#!/usr/bin/env python
"""
Notebook for chapter 7 of Data Science from Scratch.

All content copied by hand. I definitely can't claim ownership of any of this.

Usage:
    ./chapter_7.py [options]

Options:
    -h --help   Show this text.
"""

from docopt import docopt
from typing import Tuple
from chapter_6 import inverse_normal_cdf, normal_cdf
import math
import random


def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """
    Returns mu and sigma corresponding to Binomial(n, p)
    """
    mu = p * n
    sigma = math.sqrt(p * (1-p) * n)
    return mu, sigma

# probability that a variable is below some threshold
normal_probability_below = normal_cdf


def normal_probability_above(lo: float, mu: float = 0,
                             sigma: float = 1) -> float:
    """
    Probability that N(mu, sigma) is greater than lo.
    """
    return 1 - normal_cdf(lo, mu, sigma)


def normal_probability_between(lo: float, hi: float, mu: float = 0,
                               sigma: float = 1) -> float:
    """Chances that a normal distribution is between lo and hi."""
    assert lo < hi
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)


def normal_probability_outside(lo: float, hi: float, mu: float = 0,
                               sigma: float = 1) -> float:
    """Chances that a normal distribution isn't between lo & hi."""
    assert lo < hi
    return 1 - normal_probability_between(lo, hi, mu, sigma)


def normal_upper_bound(probability: float, mu: float = 0,
                       sigma: float = 1) -> float:
    """
    Finds the Z for which P(z >= Z) == probability.
    """
    return inverse_normal_cdf(probability, mu, sigma)


def normal_lower_bound(probability: float, mu: float = 0,
                       sigma: float = 1) -> float:
    """
    Finds the Z for which P(z <= Z) == probability.
    """
    return inverse_normal_cdf(1 - probability, mu, sigma)


def normal_two_sided_bounds(probability: float, mu: float = 0,
                            sigma: float = 1) -> Tuple[float, float]:
    """
    Finds symmetric bounds (about mean) for which P(Z in bounds) = probability.
    """
    tail_probability = (1 - probability) / 2

    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound


def main() -> None:
    opts = docopt(__doc__)
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
    lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)
    print(f"We found a lower bound of:   {lower_bound}\n" +
          f"  ... and an upper bound of: {upper_bound}")  # 469, 531

    # Now testing for a type 2 (false negative) probability event
    # Define the bounds at which we reject the null hypothesis (at p=0.05)
    lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

    # Consider a situation where the coin lands heads 55% of the time
    mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

    # Consider a single-tail examination of our results; what are the odds
    # a coin will land between the "fair coin" bounds if tossed? 
    type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
    # type_2_prob = 0.113: those are the odds that the result from our biased
    # coin looks like a typical result for the null hypothesis (a fair coin)

    print(f"For a biased coin toss, the odds that we'll land in the p<0.05 "
          f"bounds expected from a *fair* toss are p = {type_2_probability}.\n"
          f"Not great, but still plausible.")

    # ... what does he mean by 'power,' here? I'm slightly lost.
    # I *think* these are the odds that the null hypothesis can be rejected
    # from a test of 1,000 coin tosses, assuming 55% odds of heads.
    power = 1 - type_2_probability  # 0.887

    # Now look for the one-sided upper bound of a normal distribution for a
    # fair coin at p = 0.05
    hi = normal_upper_bound(0.95, mu_0, sigma_0)
    print(f"For a hypothesis where we expect the coin to land heads 55% of "
          f"the time, our cut-off for a one-sided test of 1,000 flips "
          f"is heads = {hi}.")

    type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
    # Still don't know why're we're calling this "power"
    power = 1 - type_2_probability  # 0.936
    print(f"The odds that we'll still get something below 536 'heads' results "
          f"is {type_2_probability}. We'll probably be able to reject the "
          "null hypothesis, but odds are too high it'll still be plausible.")

    # Now, Monte Carlo this
    extreme_value_count = 0  # test lands outside of expected range

    # Flip coin 1,000 times, count heads
    for _ in range(1000):
        num_heads = sum(1 if random.random() < 0.5 else 0
                        for _ in range(1000))  
        # If we get something that is p < 0.05 for the null hypothesis,
        # it is an extreme value, so x =< 469 or 531 <= x
        # Not sure why author is using 470 and 530.
        if num_heads >= 530 or num_heads <= 470:
            extreme_value_count += 1  # log it as an extreme result

    # p-value was 0.062, so we expect 62 extreme values out of 1000.
    # Plus or minus root(62), or 8. 
    # So outside of 54 - 70, by my count. At the one-sigma interval.
    # Not sure why the author has 59-65, which is +/- 3!
    print(f"Landed on extreme values {extreme_value_count} times.")
    # I bet this assertion fails a lot of the time.
    assert 59 < extreme_value_count < 65, f"Our simulation saw {extreme_value_count} extreme values. How unlikely!"
    


if __name__ == '__main__':
    main()
