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
