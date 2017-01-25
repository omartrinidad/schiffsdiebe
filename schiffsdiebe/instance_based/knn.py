from datasets import Examples
from utils import euclidean_distance
from collections import Counter
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


class kNN(object):
    """
    Implementation of kNN algorithm
    """


    def __init__(self, dataset):
        """
        """

        # training dataset
        self.training = dataset.training
        self.training_labels = dataset.training_labels

        # test dataset
        self.test = dataset.test
        self.test_labels = dataset.test_labels

        # matrix of distances
        self.dists = np.zeros([self.training.shape[0], self.test.shape[0]])
        self.pred_labels = {}


    def normalization(self):
        """
        Normalization of data
        """

        # get mean and standard deviation of attributes
        std = np.std(self.training, axis=0)
        mean = np.mean(self.training, axis=0)

        # normalize training and test data
        self.training = (self.training - mean)/std
        self.test = (self.test - mean)/std


    def fit(self, k=[3], metric="euclidean"):
        """
        The only thing we do is calculate the distance matrix
        """

        for i in k:
            self.pred_labels[i] = []

        # How to do this in a better way?
        for i, x in enumerate(self.test):

            if metric == "euclidean":
                self.dists[:, i] = np.sqrt(np.sum(np.subtract(x, self.training) ** 2, axis=1))
                # self.dists[:, i] = pairwise_distances(x.reshape(1, -1), self.training, metric = "euclidean")
            elif metric == "cosine":
                self.dists[:, i] = pairwise_distances(x.reshape(1, -1), self.training, metric = "cosine")

        for i in self.dists.T:
            ordered = np.argsort(i)

            # calculate value for different values of k
            for ik in k:
                indexes = ordered[0:ik]
                c = Counter(self.training_labels[indexes]).most_common(1)
                self.pred_labels[ik].append(c[0][0])


    def evaluation(self):
        """
        Calculate accuracy
        """
        for ik in self.pred_labels:
            msg = "{0} -> ks, {1:0.02f} of tests examples classified correctly".format(
                    ik, np.mean(self.pred_labels[ik] == self.test_labels) * 100
                    )
            print msg
