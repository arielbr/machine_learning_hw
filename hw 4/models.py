""" 
Keep model implementations in here.
"""

import numpy as np


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def fit(self, *, X, y, kernel_matrix):
        """ Fit the model.

        Args:
            X: A list of strings, where each string corresponds to a document.
            y: A dense array of ints with shape [num_examples].
            kernel_matrix: an ndarray containing kernel evaluations
        """
        raise NotImplementedError()

    def predict(self, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of strings, where each string corresponds to a document.
            kernel_matrix: an ndarray containing kernel evaluations

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class KernelPegasos(Model):

    def __init__(self, *, nexamples, lmbda):
        """
        Args:
            nexamples: size of example space
            lmbda: regularizer term (lambda)

        Sets:
            b: beta vector (related to alpha in dual formulation)
            t: current iteration
            kernel_degree: polynomial degree for kernel function
            support_vectors: array of support vectors
            labels_corresp_to_svs: training labels that correspond with support vectors
        """
        super().__init__(lmbda=lmbda)
        self.b = np.zeros(nexamples, dtype=int)
        self.t = 1
        self.support_vectors = None
        self.labels_corresp_to_svs = None
        self.indices = None

    def fit(self, *, X, y, kernel_matrix):
        """ Fit the model.

        Args:
            X: A list of strings, where each string corresponds to a document.
            y: A dense array of ints with shape [num_examples].
            kernel_matrix: an ndarray containing kernel evaluations
        """
        self.support_vectors = []
        self.labels_corresp_to_svs = []
        self.indices = []
        # update self.b
        num_eg = len(y)
        # convert 0's in y to -1's
        for i in range(num_eg):
            if y[i] == 0:
                y[i] = -1
        for j in range(num_eg):
            self.t += 1
            # calculate if condition
            s = 0
            for i in range(num_eg):
                s += np.dot(np.multiply(self.b[i], y[i]), kernel_matrix[i][j])
            cond = y[j] / (self.lmbda * (self.t-1)) * s
            if cond < 1:
                # update b_j
                self.b[j] += 1
        # update s_v and labels
        for i in range(len(self.b)):
            if self.b[i] > 0:
                self.indices.append(i)
                self.support_vectors.append(X[i])
                self.labels_corresp_to_svs.append(y[i])

    def predict(self, *, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of strings, where each string corresponds to a document.
            kernel_matrix: an ndarray containing kernel evaluations

        Returns:
            A dense array of ints with shape [num_examples].
        """
        alpha = self.b / (self.lmbda * (self.t-1))
        y = np.zeros(len(X))

        # update for all predictions
        for j in range(len(X)):
            pred = 0
            """for i in range(len(X)):
                pred += np.dot(np.multiply(alpha[i],
                                           self.labels_corresp_to_svs[i]), kernel_matrix[i][j])"""
            # update with svms
            for i in range(len(self.indices)):
                index = self.indices[i]
                pred += np.dot(np.multiply(alpha[index],
                                           self.labels_corresp_to_svs[i]), kernel_matrix[i][j])
            y[j] = 0 if pred < 0 else 1
        return y
