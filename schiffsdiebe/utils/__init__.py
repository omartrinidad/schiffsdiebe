# auxiliar functions

def euclidean_distance(a, b):
    """
    Calculate the Euclidean distance of two vectors
    """
    distance = 0
    for x, y in zip(a, b):
        distance += (x - y) ** 2

    return sqrt(distance)


def snake_distance(a, b):
    """
    Calculate the Manhattan distance of two vectors.
    """
    distance = 0
    for x, y in zip(a, b):
        distance += (x - y)

    return distance


def chebyshev_distance(a, b):
    """
    Calculate the Chebyshev distance of two vectors.
    """
    distances = []
    for x, y in zip(a, b):
        distances.append(abs(x - y))
    distance = max(distances)

    return distance


def entropy(feature_data):
    """
    Example playTennis entropy(["yes","yes","yes","yes","yes","yes","yes","yes","yes","no","no","no","no","no"])
    Calculates the entropy for a feature data.
    """
    if len(feature_data) == 0:
        return 1.02
    data_and_uniqe = np.unique(feature_data, return_inverse=True)
    pi = bincount(data_and_uniqe[1]).astype(np.float64)
    return np.sum(- pi/len(feature_data) * np.log2(pi/len(feature_data)))


def information_gain(sample, labels):
    """
    sample = [31, 34, 32, 20, 11, 10, 8, 23, 7, 21, 23]
    labels = [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
    # we will sort the first array, and also the second one
    # labels = [1, 1, 1, 1, 1, | 0, 0, | 1, | 0, 0, 0]
    # we calculate the information gain from each split and we return the
    # higher value
    """

    global_entropy = entropy(labels)
    cardinality = float(sample.shape[0])

    indexes = np.argsort(sample)
    flag = labels[indexes][0]

    splits_indexes = {}

    for pos, element in enumerate(labels[indexes]):

        # adjacent examples with different classes, for example: 1.0 and 0.0
        if flag != element:
            mean = (sample[indexes][pos] - sample[indexes][pos-1])/2.0
            split = sample[indexes][pos - 1] + mean
            splits_indexes[pos] = str(split)
            flag = element

    # get best split
    best_gain = ('0', 0.0)

    for index in splits_indexes:

        # < less-than split
        # index is the same as cardinality of subsets
        total =+ (index/cardinality) * entropy(labels[indexes][:index])
        # >= more-equal-than split
        total += ((cardinality - index)/cardinality) * entropy(labels[indexes][index:])

        gain = global_entropy - total

        if gain > best_gain[1]:
            best_gain = (splits_indexes[index], gain)

    # a tuple of the split value and the gain
    return best_gain
