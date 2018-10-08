#http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
#by unutbu

def argsort_list(list = None, index = 0, backwards = False):
    """Returns the sorted (alphabetically) indices of a list. Similar to argsort
    for numpy.
    """
    return sorted(
            range(len(list)), 
            key = list.__getitem__,
            reverse = backwards
        )
