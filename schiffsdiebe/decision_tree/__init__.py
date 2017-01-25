import numpy as np
from math import log
from utils import entropy, information_gain
from utils.datenstruktur import Knoten
from pprint import pprint


class tdidt(object):
    """
    Implementation of TDIDT algorithm
    """

    def __init__(self, dataset, node):
        """
        """
        # training dataset
        self.training = dataset.training
        self.training_labels = dataset.training_labels

        self.samples = self.training
        self.attributes = dataset.attributes

        # test dataset
        self.test = dataset.test
        self.test_labels = dataset.test_labels

        self.root = node


    def __tdidt(self, samples, atts, node):
        """
        """
        labels = self.training[:,-1]
        # first
        best_gain = ('', 0)

        # calculate entropy for continuous valued attributes
        for col, sample in enumerate(self.training.T[:-1]):
            # node attributes[col]
            # calculate information gain for each column
            ig = information_gain(sample, labels)

            if ig[1] > best_gain[1]:
                best_gain = (self.attributes[col], ig[1])

        left_node = Knoten()
        right_node = Knoten()

        # recursive call
        self.__tdidt(samples, atts, left_node)
        self.__tdidt(samples, atts, rigth_node)
        pass


    def train(self):
        """
        training_ds: the whole matrix
        attributes: the row with the name of each column
        """
        self.__tdidt(self.samples, self.attributes, self.root)


    def test(self, test_ds):
        """
        """
        pass
