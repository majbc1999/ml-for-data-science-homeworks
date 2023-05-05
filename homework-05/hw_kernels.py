import numpy as np
from cvxopt import matrix
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
            raise ValueError(f"Invalid input dimensions: x.shape={x.shape}," 
                             f"y.shape={y.shape}")
        
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
        Calculate the kernel function value for two floats, vectors, 
        or matrices.
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
        
        # Closed form solution: w = (K + lambda * I)^-1 * y

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
    def __init__(self, 
                 kernel, 
                 lambda_: float = 1.0, 
                 epsilon: float = 0.1,
                 tolerance: float = None) -> None:
        """
        Initialize the support vector regression model.
        """
        self.kernel = kernel
        self.C = 1 / lambda_
        self.epsilon = epsilon
        self.tolerance = 1e-3 if tolerance is None else tolerance
        self.support_vectors = None
        self.alpha = None
        self.b = None

    def fit(self, X: np.array, y: np.array):
        """
        Fit the model to the training data.
        """
        # Gram matrix
        n_samples, _ = X.shape
        K = self.kernel(X, X)

        # equation 10
        # TODO: I think the bug is here 
        P = matrix(np.outer(y, y) * K)
        P = matrix(np.vstack((np.hstack((P, -P)), np.hstack((-P, P)))))
        q = matrix(-1.0, (n_samples * 2, 1))

        # subject to 
        A = matrix(np.concatenate((y, -y)).reshape((1, n_samples * 2)), (1, n_samples * 2))
        b = matrix(0.0)
        
        # bounds for alpha
        G = matrix(np.vstack((-np.eye(n_samples * 2), np.eye(n_samples * 2))))
        h = matrix(np.hstack((np.zeros(n_samples * 2), np.ones(n_samples * 2) * self.C)))

        # call solver
        solution = qp(P, q, G, h, A, b)
        
        # extract support vectors and bias
        alpha = np.array(solution['x'])
        calculated_ys = np.array(solution['y'])

        # TODO: For some reason we should store both alpha_i and alpha_i* 
        # even though we only ever use alpha_i - alpha_i* 

        # true_alpha_i = alpha_i - alpha_i*
        true_alphas = []
        aux_list = []
        for alp in alpha:
            if len(aux_list) != 2:
                aux_list.append(alp)
            else:
                true_alphas.append(aux_list[0] - aux_list[1])
                aux_list = []
                aux_list.append(alp)
        true_alphas.append(aux_list[0] - aux_list[1])

        self.alpha = np.array(true_alphas)
        if not self.tolerance:
            self.support_vectors = X

            # calculate b with calculated ys
            self.b = np.mean(y - calculated_ys)
        else:
            self.support_vectors = X[self.alpha > self.tolerance]
            self.alpha = alpha[self.alpha > self.tolerance]
            self.b = np.mean(y - np.dot(K, self.alpha))
    
        return self

    # TODO: Correct this
    def predict(self, X):
        """
        Predict the output for new data.
        """
        # Calculate k'(x) for each x in X
        k_x = self.kernel(X, self.support_vectors)

        # Calculate the predictions
        y_pred = np.dot(k_x, self.alpha) + self.b

        return y_pred

    def get_alpha(self):
        """
        Return the alpha values.
        """
        return self.alpha


if __name__ == "__main__":
    pass