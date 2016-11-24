import numpy as np
import os

__location__ = os.path.realpath(
                    os.path.join(os.getcwd(),
                    os.path.dirname(__file__))
                    ) + "/"

class Dataset():
    """
    I want to create a similar object to that one existing in R: Dataframe
    """

    def __init__(self, training_file, test_file):
        """
        """

        self.training = np.genfromtxt(
                training_file, dtype=float,
                delimiter=',', skip_header = 1
                )[:,0:-1]

        self.training_labels = np.genfromtxt(
                training_file, dtype=float,
                delimiter=',', skip_header = 1
                )[:,-1:].ravel()

        self.test = np.genfromtxt(
                test_file, dtype=float,
                delimiter=',', skip_header = 1
                )[:,0:-1]

        self.test_labels = np.genfromtxt(
                test_file, dtype=float,
                delimiter=',', skip_header = 1
                )[:,-1:].ravel()


class Examples(object):
    """
    """

    def __init__(self):
        """
        """
        pass


    def spam(self):
        """
        Spam dataset
        """

        training_ds = __location__ + "data/spam_training.csv"
        test_ds = __location__ + "data/spam_test.csv"

        return Dataset(training_ds, test_ds)
