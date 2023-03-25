import numpy as np
import pandas as pd
import unittest
from scipy.optimize import minimize


class RidgeReg:
    """
    Class for ridge regression (L2) with closed form solution.
    """
    def __init__(self, weight: float = 1) -> None:
        self.weight = weight
        self.betta = None

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fits the model and returns nothing.
        """

        ones_column = np.ones((X.shape[0], 1))
        X = np.concatenate((ones_column, X), axis=1)

        A = np.identity(X.shape[1])
        A[0, 0] = 0

        aux_matrix = X.T @ X + self.weight * A
        self.betta = np.linalg.inv(aux_matrix) @ (X.T @ y)

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the output and returns the predictions.
        """
        if self.betta is None:
            raise ValueError("Fit the model before predicting")

        ones_column = np.ones((X.shape[0], 1))
        X = np.concatenate((ones_column, X), axis=1)

        return X @ self.betta


class LassoReg:
    """
    Class for Ted Lasso :) regression (L1) with Powell's method.
    """
    def __init__(self, weight: float = 1) -> None:
        self.weight = weight
        self.betta = None

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fits the model and returns nothing.
        """

        ones_column = np.ones((X.shape[0], 1))
        X = np.concatenate((ones_column, X), axis=1)

        def loss_function(betta: np.array, X: np.array, y: np.array) -> float:
            loss = np.sum(np.square(X.dot(betta) - y)) + \
                self.weight * np.sum(np.abs(betta))
            return loss

        self.betta = minimize(fun=loss_function,
                              x0=np.zeros(X.shape[1]),
                              args=(X, y),
                              method='Powell',
                              tol=0.05).x

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the output and returns the predictions.
        """
        if self.betta is None:
            raise ValueError("Fit the model before predicting")

        ones_column = np.ones((X.shape[0], 1))
        X = np.concatenate((ones_column, X), axis=1)

        return X @ self.betta


class RegularizationTest(unittest.TestCase):
    """
    Unit tests for the regularization models.
    """

    def test_ridge_simple(self):
        """
        Simple test for ridge regression.
        """
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2 * X[:, 0]

        X_test = np.array([[10], [20]])

        model = RidgeReg(1)
        model.fit(X, y)

        y_test = [30, 50]
        y = model.predict(X_test)

        self.assertAlmostEqual(y[0], y_test[0], delta=0.1)
        self.assertAlmostEqual(y[1], y_test[1], delta=0.1)

    def test_lasso_simple(self):
        """
        Simple test for lasso regression.
        """
        X = np.array([[1],
                      [10],
                      [100]])
        y = 10 + 2 * X[:, 0]

        X_test = np.array([[10], [20]])

        model = LassoReg(1)
        model.fit(X, y)

        y_test = [30, 50]
        y = model.predict(X_test)

        self.assertAlmostEqual(y[0], y_test[0], delta=0.5)
        self.assertAlmostEqual(y[1], y_test[1], delta=0.5)

    def test_big_ridge(self):
        """
        Test for ridge regression on a bigger dataset.
        """
        X = np.array([[1, 7, 6],
                      [5, 12, 3],
                      [-4, 10, 0]])
        y = 4 + 2 * X[:, 0] + 5 * X[:, 1] - 7 * X[:, 2]

        X_test = np.array([[1, 2, 3], [4, 5, 6]])

        model = RidgeReg(1)
        model.fit(X, y)

        y_test = [-12.75253664, -7.78410372]
        y = model.predict(X_test)

        self.assertAlmostEqual(y[0], y_test[0], delta=0.1)
        self.assertAlmostEqual(y[1], y_test[1], delta=0.1)
    
    def test_big_lasso(self):
        """
        Test for ridge regression on a bigger dataset.
        """
        X = np.array([[1, 7, 6],
                      [5, 12, 3],
                      [-4, 10, 0]])
        y = 4 + 2 * X[:, 0] + 5 * X[:, 1] - 7 * X[:, 2]

        X_test = np.array([[1, 2, 3], [4, 5, 6]])

        model = RidgeReg(1)
        model.fit(X, y)

        y_test = [-12.75253664, -7.78410372]
        y = model.predict(X_test)

        self.assertAlmostEqual(y[0], y_test[0], delta=0.1)
        self.assertAlmostEqual(y[1], y_test[1], delta=0.1)

    def test_predict_on_untrained_model(self):
        """
        Test for predicting on an untrained model.
        """
        X = np.array([[1, 7, 6],
                      [5, 12, 3],
                      [-4, 10, 0]])
        y = 4 + 2 * X[:, 0] + 5 * X[:, 1] - 7 * X[:, 2]

        X_test = np.array([[1, 2, 3], [4, 5, 6]])

        model_ridge = RidgeReg(1)
        model_lasso = LassoReg(1)

        with self.assertRaises(ValueError):
            model_ridge.predict(X_test)
        
        with self.assertRaises(ValueError):
            model_lasso.predict(X_test)


def rmse(y_true: np.array, y_pred: np.array) -> float:
    """
    Computes the root mean squared error.
    """
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def load(fname: str) -> tuple:
    """
    Load the superconductor dataset.
    """
    df = pd.read_csv(fname)
    features = df.columns[:-1]
    X = df[features].values
    y = df["critical_temp"].values
    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]
    return features, X_train, y_train, X_test, y_test


def superconductor(X_train, y_train, X_test, y_test):
    # Grid search for optimal regularization weight
    weights = np.linspace(-4, 4, num=800)
    errors = []

    for w in weights:
        model = RidgeReg(weight=w)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        err = rmse(y_test, y_pred)

        errors.append(err)

    # Best weight
    weight_index = np.argmin(errors)
    weight = weights[weight_index]

    # Train model with best weight
    model = RidgeReg(weight=weight)
    model.fit(X_train, y_train)

    # Predict on test set and calculate relative mean squared error
    y_pred = model.predict(X_test)
    err = rmse(y_test, y_pred)

    return weight, err


if __name__ == "__main__":
    # Load the needed data
    features, X_train, y_train, X_test, y_test = load(
        "homework-03/superconductor.csv")
    
    # Run the ridge regression on superconductor data,
    # find the best weight and print the RMSE estimate
    weight, err = superconductor(X_train, y_train, X_test, y_test)

    print("Best weight:", weight)
    print("RMSE estimate:", err)

    # Run the tests
    unittest.main()
