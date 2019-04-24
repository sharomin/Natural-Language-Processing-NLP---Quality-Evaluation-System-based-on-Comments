import numpy as np


class LinearClassifier():
    """ Base class for linear classifiers. """

    def __init__(self, lr=None):
        self.w = None
        self.lr = lr

    def fit(self, X, y):
        """Builds a model given the input data.
        :param X: input data. Values must be numeric.
        :param y: target variable.
        :return:
        """
        raise Exception('I cannot train...')

    def pred(self, X):
        """Predicts values for the input data given.
        :param X: input data.
        """
        X = self.preprocess_input(X)
        return X.dot(self.w)

    def preprocess_input(self, X, y=None):
        X = X.reshape(X.shape[0], -1)
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((X, bias))
        if y is None:
            return X
        y = y if len(y.shape) > 1 else y[:, np.newaxis]
        return X, y


class LinearRegressionMSE(LinearClassifier):
    """Linear regression using closed-form solution."""

    def fit(self, X, y):
        X, y = self.preprocess_input(X, y)
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
        return self


class LinearRegressionGD(LinearClassifier):
    """Linear regression using gradient descent."""

    def __init__(self, lr):
        super().__init__(lr)
        self.step = 0

    def fit(self, X, y, tol=10e-9):
        X, y = self.preprocess_input(X, y)
        # init value of weights
        w_ = np.random.rand(X.shape[1], 1)
        self.w = np.ones((X.shape[1], 1))
        error = 1

        while (error > tol):
            dw = 2 * (X.T.dot(X).dot(w_) - X.T.dot(y))
            w_ -= self.lr.compute(dw)

            error = np.linalg.norm(np.abs(self.w - w_), 2)
            self.w = w_.copy()
            self.step += 1
        return self
