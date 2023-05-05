import numpy as np
from cvxopt.solvers import qp
from typing import Union

    
class RBF:
    """
    Radial basis function kernel
    """
    def __init__(self, sigma: float = 1.0):
        """
        Initialize the kernel with a sigma value.
        """
        self.sigma = sigma
    
    def __call__(self, x: np.array, y: np.array) -> np.array:
        """
        Calculate the kernel function value for two vectors or matrices.
        """
        # Check the dimensions of the input arrays
        x_dim = x.ndim
        y_dim = y.ndim
        
        # Two 1D vectors
        if x_dim == 1 and y_dim == 1:
            distance = np.linalg.norm(x - y)
            kernel = np.exp(-distance ** 2 / (2 * self.sigma ** 2))
            
        # 1D vector and 2D matrix
        elif x_dim == 1 and y_dim == 2:
            distance = np.linalg.norm(x - y, axis=1)
            kernel = np.exp(-distance ** 2 / (2 * self.sigma ** 2))
            
        # 2D matrix and 1D vector
        elif x_dim == 2 and y_dim == 1:
            distance = np.linalg.norm(x - y, axis=1)
            kernel = np.exp(-distance ** 2 / (2 * self.sigma ** 2))
            
        # Two 2D matrices
        elif x_dim == 2 and y_dim == 2:
            x_norm = np.sum(x ** 2, axis=1).reshape(-1, 1)
            y_norm = np.sum(y ** 2, axis=1).reshape(1, -1)
            # formula in tests
            distance = np.sqrt(x_norm + y_norm - 2 * np.dot(x, y.T))
            kernel = np.exp(-distance ** 2 / (2 * self.sigma ** 2))
            
        # Invalid input dimensions
        else:
            raise ValueError(f"Invalid input dimensions: x.shape={x.shape}, y.shape={y.shape}")
        
        return kernel


class Polynomial:
    """
    Polynomial kernel
    """
    def __init__(self, M: int = 2):
        """
        Initialize the kernel with a M value.
        """
        self.M = M

    def __call__(self, x: np.array, y: np.array) -> float:
        """
        Calculate the kernel function value for two floats, vectors, or matrices.
        """
        
        # Check the dimensions of the input arrays
        x_dim = x.ndim
        y_dim = y.ndim
        
        # Two 1D vectors
        if x_dim == 1 and y_dim == 1:
            kernel = (1 + np.dot(x, y)) ** self.M
            
        # 1D vector and 2D matrix
        elif x_dim == 1 and y_dim == 2:
            kernel = (1 + np.dot(x, y.T)) ** self.M
            kernel = kernel.squeeze()  # Remove singleton dimension
            
        # 2D matrix and 1D vector
        elif x_dim == 2 and y_dim == 1:
            kernel = (1 + np.dot(x, y)) ** self.M
            kernel = kernel.squeeze()  # Remove singleton dimension
            
        # Two 2D matrices
        elif x_dim == 2 and y_dim == 2:
            kernel = (1 + np.dot(x, y.T)) ** self.M
            
        # Invalid input dimensions
        else:
            raise ValueError(f"Invalid input dimensions:" 
                             f"x.shape={x.shape}, y.shape={y.shape}")
        
        return kernel


class KernelizedRidgeRegression:
    """
    Class for kernelized ridge regression.
    """
    def __init__(self, kernel, lambda_: float = 1.0) -> None:
        """
        Initialize the kernelized ridge regression model.
        """
        self.kernel = kernel
        self.lambda_ = lambda_

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fit the model to the training data.
        """
        # Calculate the kernel matrix, K
        K = self.kernel(X, X)
        
        # Closed form solution
        # w = (K + lambda * I)^-1 * y

        # Calculate the inverse of the kernel matrix
        K_inv = np.linalg.inv(K + self.lambda_ * np.eye(K.shape[0]))
        
        # Save x-es for prediction
        self.X = X

        # Precompute (K + lambda * I)^-1 * y
        self.w = np.dot(K_inv, y)

        return self
    
    def predict(self, X: np.array) -> np.array:
        """
        Predict the output for new data.
        """
        # Calculate k'(x) for each x in X
        k_x = self.kernel(X, self.X)

        # Calculate the predictions
        y_pred = np.dot(k_x, self.w)

        return y_pred


class SVR:
    """
    Class for support vector regression.
    """





if __name__ == "__main__":
    pass