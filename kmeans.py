#!/usr/bin/python
# encoding: utf8

# fix this:
# kmeans.py:95: RuntimeWarning: invalid value encountered in divide
# total = total/self.dataset[self.clusters[c]].shape[0]
# kmeans.py:115: RuntimeWarning: invalid value encountered in divide
#  self.parent.title("Not convergence")

from numpy import genfromtxt, where, random, savetxt
import numpy as np
from math import sqrt
from Tkinter import Tk, Canvas, Frame, BOTH
from pprint import pprint


def snake_distance(a, b):
    """
    Calculate the Manhattan distance of two vectors.
    """
    distance = 0
    for x, y in zip(a, b):
        distance += (x - y)

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

    def __init__(self, dataset, k):
        """
        """
        self.dataset = np.genfromtxt(
                    dataset, dtype=float,
                    delimiter=',', usecols = (0, 1, 2)
                    )

        # this depends on the number of columns in the dataset
        self.no_dimensions = self.dataset.shape[1]
        self.k = k

        # initialize centroids
        centroids = np.zeros((k, self.no_dimensions))
        for i in range(k):
            for j, d in enumerate(self.dataset.transpose()):
                mini, maxi = min(d), max(d)
                centroids[i, j] = np.random.uniform(mini, maxi)

        self.centroids = centroids
        self.clusters = {}

        # flag variable to control the convergence
        self.convergence = False

        self.next()


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
                print len(intersection),
                if len(intersection) > maxi:
                    maxi = len(intersection)

            print maxi
            total += maxi

        purity_value = 1.0/N * total


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
                    closest.append(euclidean_distance(centroid, element))

                # dirty code +_+
                index = closest.index(min(closest))
                clusters[index].append(i)

            self.clusters = clusters

            # recalculate centroids
            old_centroids = self.centroids.copy()

            for c in self.clusters:
                # get the sum of each column (dimension)
                total = np.sum(self.dataset[self.clusters[c]], axis=0)
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
            self.parent.title("Convergence reached")

            self.kmeans.purity_function()

            self.after(delay)

def main():
    root = Tk()
    kmeans = KMeans('iris.data', 3)
    #kmeans = KMeans('shikis.data', 3)
    ex = Example(root, 900, kmeans)
    root.mainloop()

if __name__ == '__main__':
    main()
