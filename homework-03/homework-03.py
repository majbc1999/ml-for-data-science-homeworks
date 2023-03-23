import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, Lasso
# NOTE: sklearn is only used for testing the models


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
        aux_matrix = X.T @ X + self.weight * np.identity(X.shape[1])
        self.betta = np.linalg.inv(aux_matrix) @ X.T @ y

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the output and returns the predictions.
        """
        if self.betta is None:
            raise ValueError("Fit the model before predicting")

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
        def loss_function(betta: np.array) -> float:
            loss =  np.sum(np.square(X.dot(betta) - y)) + \
                        self.weight * np.sum(np.abs(betta))
            return loss
        
        self.betta = minimize(fun=loss_function, 
                              x0=np.zeros(X.shape[1]), 
                              method='Powell').x

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the output and returns the predictions.
        """
        if self.betta is None:
            raise ValueError("Fit the model before predicting")
        return X @ self.betta


def test_ridge_reg():
    """
    Test correct output with help of sklearn Ridge.
    """
    # create toy data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([5, 7, 9])
    weight = 0.5

    # initialize model
    model = RidgeReg(weight)
    model_2 = Ridge(alpha=weight, solver="cholesky")

    # fit model
    model.fit(X, y)
    model_2.fit(X, y)

    # check bettas
    print(f'RidgeReg bettas: {model.betta}')
    print(f'True ridge bettas: {model_2.coef_}')

    # check prediction
    X_test = np.array([[17, 18]])
    y_pred = model.predict(X_test)
    y_pred2 = model_2.predict(X_test)
    print(f'RidgeReg prediction: {y_pred}')
    print(f'True ridge prediction: {y_pred2}')


def test_lasso_reg():
    """
    Test correct output with help of sklearn Lasso.
    """
    # create toy data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([5, 7, 9])
    weight = 0.5

    # initialize model
    model = LassoReg(weight)
    model_2 = Lasso(alpha=weight)

    # fit model
    model.fit(X, y)
    model_2.fit(X, y)

    # check bettas
    print(f'LassoReg bettas: {model.betta}')
    print(f'True lasso bettas: {model_2.coef_}')

    # check prediction
    X_test = np.array([[17, 18]])
    y_pred = model.predict(X_test)
    y_pred2 = model_2.predict(X_test)
    print(f'LassoReg prediction: {y_pred}')
    print(f'True lasso prediction: {y_pred2}')


if __name__ == '__main__':
    # Test ridge regression
    test_ridge_reg()

    # Test Lasso regression
    test_lasso_reg()
