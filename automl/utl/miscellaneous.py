"""Module to define useful but no AutoMl-specific methods."""

import random
import string


def argsort_list(src_list=None, backwards=False):
    """Return the sorted (alphabetically) indices of a src_list.

    Similar to argsort for numpy arrays. It is based on `unutbu's answer in
    StackOverflow <http://stackoverflow.com/questions/3382352/equivalent-of-
    numpy-argsort-in-basic-python/3382369#3382369>`_.

    Args:
        src_list    (list): The list to sort. Defaults to None.
        backwards   (bool): Whether or not to do reverse sort. Defaults to
            False.

    Returns:
        list: The resulting list of indices sorted by value.

    """
    return sorted(range(len(src_list)), key=src_list.__getitem__,
                  reverse=backwards)


def generate_random_id(n_chars=6):
    """Generate an internal ID of a given length.

    Args:
        n_chars (int): The length. Defaults to 6.

    Returns:
        str: The resulting ID.

    """
    return ''.join(
        random.choice(
            string.ascii_uppercase + string.digits
        ) for _ in range(n_chars)
    )
