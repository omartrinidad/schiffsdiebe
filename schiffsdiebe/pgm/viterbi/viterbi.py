#!/usr/bin/python
# encoding: utf8

import numpy as np
from math import log

class HMM():
    """
    Very good tutorial: https://web.stanford.edu/~jurafsky/slp3/8.pdf
    One of the applications of HMM,  is to predict the sequence of state
    changes, based on the sequence of observations

    A remark on Bayes Probability: If A and B are independent events:
    Pr(A|B) = Pr(A)
    """

    def __init__(self):
        """
        """
        self.sequence = "GGCACTGAA"
        self.hidden = []
        self.hidden2 = None
        self.states = []
        self.connectors = None
        self.table_probabilities = None

    def init_hidden_state(self, total):
        """
        The hidden states
        """
        self.hidden2 = np.random.rand(total)
        for t in range(total):
            value = {"value": np.random.uniform()}
            self.hidden.append(value)

    def create_state(self, a=0, c=0, g=0, t=0):
        """
        """
        values = {
                "A": log(a, 2), "C": log(c, 2),
                "G": log(g, 2), "T": log(t, 2)
                }
        self.states.append(values)

    def init_connectors(self):
        # init the connectors
        a, b = self.hidden2.size, len(self.states)
        self.connectors = np.zeros((a, b)) + 0.5
        self.connectors = np.log2(self.connectors, self.connectors)

    def compute_probabilities(self):
        """
        """
        #print self.sequence
        a, b = len(self.states), len(self.sequence)
        table = np.zeros((a, b))
        summation = 0

        for i, element in enumerate(self.sequence):

            previous = table[:,i-1].max() if i > 0 else 0

            for j, state in enumerate(self.states):
                summation += -1 + state[element] + previous
                table[j, i] = summation
                summation = 0

        self.table_probabilities = table
        print self.table_probabilities


hmm = HMM()
hmm.init_hidden_state(1)
hmm.create_state(a=0.2, c=0.3, g=0.3, t=0.2)
hmm.create_state(a=0.3, c=0.2, g=0.2, t=0.3)
hmm.init_connectors()
hmm.compute_probabilities()
