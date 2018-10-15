"""Module to define useful but not specific methods."""


def argsort_list(src_list=None, backwards=False):
    """Return the sorted (alphabetically) indices of a src_list.

    Similar to argsort for numpy arrays. It is base on unutbu's answer in
    StackOverflow:
        http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort
        -in-basic-python/3382369#3382369

    Attributes:
        src_list        (src_list) The src_list to sort. Default is None.
        backwards   (bool) Whether to do reverse sort. Default is False.

    Returns:
        src_list: The resulting array of indices sorted by value.

    """
    return sorted(range(len(src_list)), key=src_list.__getitem__,
                  reverse=backwards)
