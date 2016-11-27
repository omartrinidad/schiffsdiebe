import numpy as np
from math import log
from utils import entropy, information_gain
from utils.datenstruktur import Knoten
from pprint import pprint


class DecisionTree(object):
    """
    Implementation of TDIDT algorithm
    """

    def __init__(self, training_ds, attibutes, node):
        """
        """
        self.root = node
        self.dataset = training_ds
        self.attributes = attributes


    def __tdidt(self, samples, atts, node):
        """
        """
        labels = training_ds[:,-1]
        # first
        best_gain = ('', 0)

        # calculate entropy for continuous valued attributes
        for col, sample in enumerate(training_ds.T[:-1]):
            # node attributes[col]
            # calculate information gain for each column
            ig = inf_gain(sample, labels)

            if ig[1] > best_gain[1]:
                best_gain = (attributes[col], ig[1])

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
        self.__tdidt(self.dataset, self.attributes, self.root)


    def test(self, test_ds):
        """
        """
        pass


# get the X columns one by one and do the comparison with the column y
root = Knoten()
tree_gene_expression = DecisionTree(training_ds, attributes, root)
tree_gene_expression.train()
tree_gene_expression.test(test_ds)

tree = DecisionTree()
