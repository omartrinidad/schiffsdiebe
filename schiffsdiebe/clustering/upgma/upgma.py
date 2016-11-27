from math import sqrt
import numpy as np

# nice trick numpy
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def euclidean_distance(a, b):
    """
    Calculate the Euclidean distance of two vectors
    """
    distance = 0
    for x, y in zip(a, b):
        distance += (x - y) ** 2

    return sqrt(distance)

def get_distance_matrix(points):
    """
    From a group of points calculate the distance matrix between them
    """
    symbols = "ABCDEFGHIJKLMNOP"

    w = h = len(points)
    distance_matrix = np.ndarray(shape=(w, h))

    print ' ' * 5,
    for x in range(len(points)):
        print ("A%i" % (x+1)).ljust(5, ' '),
    print

    for x, p in enumerate(points):
        print ("A%i" % (x+1)).ljust(5, ' '),
        for y, q in enumerate(points):
            d = euclidean_distance(p, q)
            print ("%.2f" % d).ljust(5, ' '),
            distance_matrix[x, y] = d
        print

    return distance_matrix


points = [
            [0, 10], [2, 2], [2, 4],
            [8, 8], [7, 6], [10, 9]
        ]

points = [
            [2, 10], [5, 8], [1, 2], [2, 5],
            [8, 4], [7, 5], [6, 4], [4, 9]
        ]

distance_matrix = get_distance_matrix(points)
