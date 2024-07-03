#!/usr/bin/env python
"""
Going through chapter 1

Usage:
    ./foo.py [options]

Options:
    -h --help   Show this text.
"""

from docopt import docopt
from collections import Counter, defaultdict


# Values copy/pasted from book
users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" }]
friendship_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                    (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]
interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")]


# Initialize the dict with an empty list for each user:
friendships = {user["id"]: [] for user in users}
# Loop over the friendship pairs to populate them:
for i, j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)
# Make a mapping of user_id to interests
interests_by_user_id = defaultdict(list)
for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)
# And a map of interests to user_id
user_ids_by_interest = defaultdict(list)
for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)


def number_of_friends(user):
    """How many friends does a user have?"""
    user_id = user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)


def find_friends_of_friends_v0(user):
    """
    Finds the friends of a user's friends.

    This is a bad version, do not use.
    """
    return [friend_of_friend_id
            for friend_id in friendships[user["id"]]
            for friend_of_friend_id in friendships[friend_id]]


def find_friends_of_friends(user):
    """
    Finds the friends of friends of the user.

    Includes only friends the user doesn't know (and who aren't the user).
    """
    user_id = user["id"]
    return Counter(
        friend_of_friend_id
            for friend_id in friendships[user_id]
            for friend_of_friend_id in friendships[friend_id]
            if friend_of_friend_id != user_id
            and friend_of_friend_id not in friendships[user_id])


def find_data_scientists_who_like_v0(target_interest):
    """
    Finds the ids of users who like the target interest.
    """
    return [user_id
                for user_id, user_interest in interests
                if user_interest == target_interest]


def most_common_interests_with(user):
    return Counter(
        interested_user_id
            for interest in interests_by_user_id[user["id"]]
            for interested_user_id in user_ids_by_interest[interest]
            if interested_user_id != user["id"])


def main() -> None:
    opts = docopt(__doc__)

    total_connections = sum(number_of_friends(user) for user in users)

    num_users = len(users)
    avg_connections = total_connections / num_users

    num_friends_by_id = [(user["id"], number_of_friends(user))
                         for user in users]
    # Sort by number of friends, descending
    num_friends_by_id.sort(key=lambda id_and_friends: id_and_friends[1],
                           reverse=True)

    print(f"Total number of connections is {total_connections}")
    print(f"Average number of connections is {total_connections / num_users}")
    print(f"Sorted list of friends by id: {num_friends_by_id}")
    print(f"Bad list of friends of friends of user_id = 0: {find_friends_of_friends_v0(users[0])}")
    print(f"Better list of friends of friends of user_id = 3: {find_friends_of_friends(users[3])}")
    print(f"All user ID's of users who like Python: {find_data_scientists_who_like_v0('Python')}")
    print(f"Users who like each interest of user 2, and the number of interests they have in common: {most_common_interests_with(users[2])}")


if __name__ == '__main__':
    main()
