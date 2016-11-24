import numpy as np
from math import sqrt

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

dataset = np.genfromtxt(
            "dummy.data", dtype=float,
            delimiter=',', usecols = (0, 1)
            )

def gdbscan(set_of_points, n_pred, min_card, w_card):
    """
    Do the clustering with GDBSCAN algorithm proposed by
    set_of_points,
    n_pred,
    min_card,
    w_card,
    """

    # print set_of_points
    # create a group of indexes
    is_classified = np.zeros((dataset.shape[0], 1), dtype=bool)
    # print is_classified
    noise = np.zeros((1, is_classified.shape[1]))
    # print noise
    cluster = {}

    # cluster_id = next_id(noise), why this?
    cluster_id = 0

    for point, classified in zip(set_of_points, is_classified):
        if not classified:
            if expand_cluster(
                    set_of_points, point, cluster_id,
                    n_pred, min_card, w_card
                    ):
                cluster_id = cluster_id + 1


def expand_cluster(set_of_points, point, cluster_id, n_pred, min_card, w_card):
    pass


def w_card(points):
    return len(points)

def neighborhood(index, epsilon):
    distances = get_distance_matrix(dataset)[index]
    return np.where(distances < epsilon)[0]

min_card = 4
n_pred = 3
gdbscan(dataset, n_pred, min_card, w_card)
