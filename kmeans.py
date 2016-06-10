#!/usr/bin/python
# encoding: utf8

from numpy import genfromtxt, where, random, savetxt
import numpy as np
from math import sqrt
from Tkinter import Tk, Canvas, Frame, BOTH, Label, LabelFrame
from pprint import pprint


def generate_gaussian_points(mini, maxi):
    """
    Function to generate 3 centroids and a dataset with Gaussian distribution
    based on those centroids
    """
    k, dimensions = 3, 2
    centroids = np.zeros((k, dimensions))
    dataset = np.zeros((200, dimensions))

    for i in range(k):
        for j in range(dimensions):
            centroids[i, j] = np.random.uniform(mini, maxi)

    sigma_squared = np.random.uniform(mini, maxi)

    # generate 200 points
    for i in range(200):
        # select one centroid
        selected = centroids[np.random.randint(0, 3)]
        # np.random.normal is an automagical function to generate a random
        # point with normal distribution
        dataset[i] = np.random.normal(selected, sigma_squared, 2)

    return dataset, centroids


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


def euclidean_distance(a, b):
    """
    Calculate the Euclidean distance of two vectors
    """
    distance = 0
    for x, y in zip(a, b):
        distance += (x - y) ** 2

    return sqrt(distance)


class KMeans(object):
    """
    Kmeans implementation
    """

    def __init__(self, dataset, centroids, distance="euclidean"):
        """
        """

        if distance == "euclidean":
            self.distance = euclidean_distance
        elif distance == "shebyshev":
            self.distance = chebyshev_distance
        else:
            print 'Warning: Unknown distance, Euclidean distance will be used'
            self.distance = euclidean_distance

        self.dataset = dataset
        self.centroids = centroids
        self.clusters = {}

        # this depends on the number of columns in the dataset
        self.no_dimensions = self.dataset.shape[1]
        self.k = centroids.shape[0]

        # flag variable to control the convergence
        self.convergence = False

        #self.next()


    def purity_function(self):
        """
        """

        intersect = lambda x, y: list(set(x) & set(y))

        # ToDo. Group labels in a correct way
        Ks = {
                0: range(50),
                1: range(50, 100),
                2: range(100, 150)
             }

        N = self.dataset.shape[0]

        total = 0
        for k in Ks:
            maxi = 0
            for c in self.clusters:
                intersection = intersect(Ks[k], self.clusters[c])
                if len(intersection) > maxi:
                    maxi = len(intersection)

            total += maxi

        purity_value = 1.0/N * total
        return purity_value


    def next(self):
        """
        Each step of the algorithm is depicted here!
        """

        if not self.convergence:

            # assign each instance to closest center
            clusters = {}
            for i in range(self.k):
                clusters[i] = []

            for i, element in enumerate(self.dataset):
                closest = []
                for centroid in self.centroids:
                    closest.append(self.distance(centroid, element))

                # dirty code +_+
                index = closest.index(min(closest))
                clusters[index].append(i)

            self.clusters = clusters

            # recalculate centroids
            old_centroids = self.centroids.copy()

            for c in self.clusters:
                # get the sum of each column (dimension)
                total = np.sum(self.dataset[self.clusters[c]], axis=0)

                #print
                #print self.dataset[self.clusters[c]]
                #print self.dataset[self.clusters[c]].shape[0]

                total = total/self.dataset[self.clusters[c]].shape[0]

                # update the centroids
                self.centroids[c] = total

            # expensive :O
            comparison = old_centroids == self.centroids
            self.convergence = np.sum(comparison) == comparison.size


class Example(Frame):

    def __init__(self, parent, w_size, kmeans):
        Frame.__init__(self, parent)
        self.parent = parent
        self.pack(fill=BOTH, expand=1)
        self.title = self.parent.title()
        self.w_size = w_size

        # canvas
        self.canvas = Canvas(self, bg="#000")
        self.parent.title("Not convergence")
        self.canvas.pack(side="top", fill="both", expand="true")

        self.kmeans = kmeans
        self.draw(200)


    def draw(self, delay):
        """
        """
        colors = ["#0f0", "#0ff", "#ff0", "#fff"]

        self.title = self.parent.title()

        width, height = self.w_size, self.w_size
        self.canvas.config(width=width, height=height)
        self.canvas.delete("all")

        min, max = -2, 10
        range_data = max - min
        step = float(self.w_size)/range_data

        # draw centroids
        for k, c in enumerate(self.kmeans.centroids):
            x, y = (max - c[0])*step , (max - c[1])*step
            x, y = self.w_size - x, self.w_size - y

            self.canvas.create_rectangle(
                    x, y, x+15, y+15,
                    fill=colors[k]
                    )

        # draw clusters
        for k in self.kmeans.clusters:
            for i in self.kmeans.clusters[k]:

                row = self.kmeans.dataset[i]
                x, y = (max - row[0])*step , (max - row[1])*step
                x, y = self.w_size - x, self.w_size - y

                self.canvas.create_oval(
                        x, y, x+3, y+3,
                        outline=colors[k], fill=colors[k]
                        )

        self.kmeans.next()

        if not self.kmeans.convergence:
            self.after(delay, lambda: self.draw(delay))
        else:

            text = self.kmeans.purity_function()
            self.parent.title("Convergence reached - Purity value %s" % text)

            self.after(delay)

def main():

    dataset = np.genfromtxt(
                "iris.data", dtype=float,
                delimiter=',', usecols = (0, 1, 2, 3)
                )

    # initialize centroids
    no_dimensions = dataset.shape[1]
    k = 3

    centroids = np.zeros((k, no_dimensions))
    for i in range(k):
        for j, d in enumerate(dataset.transpose()):
            mini, maxi = min(d), max(d)
            centroids[i, j] = np.random.uniform(mini, maxi)

    root = Tk()

    kmeans = KMeans(dataset, centroids, distance = "shebyshev")
    # kmeans = KMeans(dataset, centroids, distance = "euclidean")

    ex = Example(root, 900, kmeans)
    root.mainloop()


def main2():
    dataset, centroids = generate_gaussian_points(1, 200)
    kmeans = KMeans(dataset, centroids)

    # results for the computational experiments
    results = open("results_experiment.txt", "w+")
    for i in range(20):

        dataset, centroids = generate_gaussian_points(1, 200)

        results.write("\n\niteration %i" % (i + 1))
        results.write("\noriginal centroids\n")
        results.write(str(centroids))

        kmeans = KMeans(dataset, centroids, distance = "euclidean")
        while not kmeans.convergence:
            kmeans.next()
        results.write("\nfinal centroids [euclidean]\n")
        results.write(str(kmeans.centroids))

        kmeans = KMeans(dataset, centroids, distance = "shebyshev")
        while not kmeans.convergence:
            kmeans.next()
        results.write("\nfinal centroids [shebyshev]\n")
        results.write(str(kmeans.centroids))

    results.close()

if __name__ == '__main__':
    # main <-- experiment with Iris dataset
    main()
    # main2 <-- Gaussian experiment
    #main2()

