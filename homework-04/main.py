import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
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
        # Add a column of ones for the intercept
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        # Initialize coefficients
        self.coef = np.zeros((self.n_classes, X.shape[1]))
        # Optimization bounds
        bounds = [(None, None) for _ in range(self.coef.size)]

        # Run gradient descent
        new_coef = fmin_l_bfgs_b(self._neg_log_likelihood, 
                                 self.coef.flatten(), 
                                 args=(X, y), 
                                 bounds=bounds,
                                 approx_grad=True,
                                 factr=10)[0]
        # Reshape and store coefficients
        self.coef = new_coef.reshape(self.n_classes, X.shape[1])
        return self

    def predict(self, X):
        # Check if model is built
        if self.coef is None:
            raise Exception("Model not yet built")
        
        # Add a column of ones for the intercept
        X = np.hstack([X, np.ones((X.shape[0], 1))])
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
        self.n_classes = None
        self.coef = None    # coefficients
        self.t = None       # thresholds
        self._deltas = None # threshold deltas

    def _deltas_to_thresholds(self, deltas: np.ndarray) -> np.ndarray:
        """
        Converts threshold deltas to thresholds.
        """
        t = ['-infty', 0]
        for delta in deltas:
            t.append(t[-1] + delta)
        t.append('infty')

        # Check if t is of correct size
        if len(t) != self.n_classes + 1:
            raise Exception("Invalid number of thresholds")
        return t

    def _neg_log_likelihood(self, coef_and_deltas: np.ndarray,
                                  X: np.ndarray, 
                                  y: np.ndarray) -> float:
        """
        Returns the log likelihood of the model.
        """
        # Reshape coefficients and thresholds

        coef = coef_and_deltas[:self.coef.size].reshape(self.coef.shape)
        deltas = coef_and_deltas[self.coef.size:]

        # Check length of deltas
        if len(deltas) != self.n_classes - 2:
            raise Exception("Invalid number of deltas")

        # Convert deltas to thresholds
        t = self._deltas_to_thresholds(deltas)

        # Predicted probabilities
        predictions = []
        for x in X:
            predictions.append(
                self._predict_single_sample(x, t, coef)
            )
        probabilities = np.array(predictions)

        # Compute negative log-likelihood
        neg_log_likelihood = 0
        for i in range(X.shape[0]):
            neg_log_likelihood -= np.log(probabilities[i, y[i]])
        return neg_log_likelihood

    def build(self, X: np.array, y: np.array):
        """
        Builds the model.
        """
        # Set number of classes
        self.n_classes = len(np.unique(y))

        # Add a column of ones for the intercept
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        # Initialize coefficients
        self.coef = np.zeros((self.n_classes - 1, X.shape[1]))
        
        # Initialize deltas for thresholds
        self._deltas = np.zeros(self.n_classes - 2)

        # Optimization bounds
        bounds = [(None, None) for _ in range(self.coef.size)] + \
                    [(0.001, None) for _ in range(self._deltas.size)]

        # Concatenate coefficients and deltas
        coef_and_deltas = np.concatenate((self.coef.flatten(), self._deltas))

        # Run gradient descent
        new_coef_and_deltas = fmin_l_bfgs_b(self._neg_log_likelihood,
                                            coef_and_deltas,
                                            args=(X, y),
                                            bounds=bounds)[0]

        # Extract coefficients and deltas
        self.coef = new_coef_and_deltas[:self.coef.size].reshape(self.coef.shape)  # noqa:501
        self._deltas = coef_and_deltas[self.coef.size:]

        # Check length of deltas
        if len(self._deltas) != self.n_classes - 2:
            raise Exception("Invalid number of deltas")

        # Convert deltas to thresholds
        self.t = self._deltas_to_thresholds(self._deltas)
    
        # Check length of thresholds
        if len(self.t) != self.n_classes + 1:
            raise Exception("Invalid number of thresholds")

        # Return self
        return self
        
    def _CDF(self, x: Union[float, str]) -> float:
        """
        CDF of the standard logistic distribution.
        """
        if x == 'infty':
            return 1
        elif x == '-infty':
            return 0
        else:
            return 1 / (1 + np.exp(-x))

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the probabilities of each class.
        """
        # Check if model is built
        if self.coef is None:
            raise Exception("Model not yet built")
    
        # Add a column of ones for the intercept
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        # Iterate through samples
        predictions = []
        for x in X:
            predictions.append(
                self._predict_single_sample(x, self.t, self.coef)
            )
        return np.array(predictions)

    def _predict_single_sample(self, 
                              x: np.array,
                              t: np.array,
                              coef: np.array) -> np.array:
        """
        Predicts the probabilities of each class.
        """
        # Check if model is built
        
        if coef is None:
            raise Exception("Model not yet built")
        
        else:
            probs = []
            for i in range(self.n_classes):
                u_i = x @ coef

                first = t[i + 1] - u_i if t[i + 1] != 'infty' else 'infty'
                second = t[i] - u_i if t[i] != '-infty' else '-infty'

                probs.append(
                    self._CDF(first) - self._CDF(second)
                )
            return np.array(probs)


def multinomial_bad_ordinal_good():
    return
 
MBOG_TRAIN = 1000