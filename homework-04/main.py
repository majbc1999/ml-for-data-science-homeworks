import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fmin_l_bfgs_b

# PART I: models

# multinomial logistic regression
class MultinomialLogReg:
    """
    Multinomial logistic regression model.
    """
    def __init__(self) -> None:
        """
        Initialize the model.
        """
        self.n_classes = None
        self.coef = None

    def _softmax(self, u: np.ndarray) -> np.ndarray:
        """
        Returns the softmax of z.
        """
        # Make sure that z has lowest possible values for numerical stability
        u -= np.min(u)
        # Compute softmax
        return (np.exp(u).T / np.sum(np.exp(u), axis=1)).T

    def _neg_log_likelihood(self, coef: np.ndarray,
                                  X: np.ndarray, 
                                  y: np.ndarray) -> float:
        """
        Returns the log likelihood of the model.
        """
        coef = coef.reshape((self.n_classes, X.shape[1]))
        # Predicted probabilities
        probabilities = self._softmax(X @ coef.T)
        # Compute negative log-likelihood
        neg_log_likelihood = 0
        for i in range(X.shape[0]):
            neg_log_likelihood -= np.log(probabilities[i, y[i]])
        return neg_log_likelihood

    def build(self, X: np.ndarray, y: np.ndarray):
        """
        Builds the model.
        """
        # Set number of classes
        self.n_classes = len(np.unique(y))
        # Initialize coefficients
        self.coef = np.zeros((self.n_classes, X.shape[1]))
        # Optimization bounds
        bounds = [(-1, 1) for _ in range(self.coef.size)]

        # Run gradient descent
        new_coef = fmin_l_bfgs_b(self._neg_log_likelihood, 
                                 self.coef.flatten(), 
                                 args=(X, y), 
                                 bounds=bounds, 
                                 approx_grad=True)[0]
        # Reshape and store coefficients
        self.coef = new_coef.reshape(self.n_classes, X.shape[1])
        return self


    def predict(self, X):
        # Check if model is built
        if self.coef is None:
            raise Exception("Model not yet built")
        
        return self._softmax(X @ self.coef.T)


# ordinal logistic regression
class OrdinalLogReg:
    """
    Ordinal logistic regression model.
    """

    def __init__(self) -> None:
        """
        Initialize the model.
        """
        self.coef = None

    def build(self, X: np.array, y: np.array):
        """
        Builds the model.
        """
        return
        

def multinomial_bad_ordinal_good():
    return

MBOG_TRAIN = 1000