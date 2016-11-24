import numpy as np
from math import log
from functions import entropy, inf_gain
from Datenstruktur import Knoten
from pprint import pprint


def information_gain(sample, labels):
    """
    sample = [31, 34, 32, 20, 11, 10, 8, 23, 7, 21, 23]
    labels = [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]

    # we will sort the first array, and also the second one
    # labels = [1, 1, 1, 1, 1, --SPLIT-- 0, 0, --SPLIT-- 1, --SPLIT-- 0, 0, 0]

    # we calculate the information gain from each split and we return the higher value

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
        break


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


attributes = np.genfromtxt(
            "dummy.csv", dtype=str,
            delimiter=',', max_rows = 1
            )

training_ds = np.genfromtxt(
            "dummy.csv", dtype=float,
            delimiter=',', skip_header = 1
            )

test_ds = np.genfromtxt(
            "gene_expression_test.csv", dtype=float,
            delimiter=',', skip_header = 1
            )

# get the X columns one by one and do the comparison with the column y
root = Knoten()
tree_gene_expression = DecisionTree(training_ds, attributes, root)
tree_gene_expression.train()
tree_gene_expression.test(test_ds)
