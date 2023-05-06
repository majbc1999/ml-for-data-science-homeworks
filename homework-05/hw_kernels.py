import numpy as np
import cvxopt
from cvxopt import matrix
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
                 epsilon: float = 0.1) -> None:
        """
        Initialize the support vector regression model.
        """
        self.kernel = kernel
        self.C = 1 / lambda_
        self.epsilon = epsilon
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
        matrix_p = []
        for i in range(n_samples):
            aux = []
            for j in range(n_samples):
                aux.append(K[i, j])
                aux.append(-K[i, j])
            matrix_p.append(aux)
            matrix_p.append([-x for x in aux])

        vec_q = []
        for i in range(n_samples):
            vec_q.append(self.epsilon - y[i])
            vec_q.append(self.epsilon + y[i])

        P = matrix(matrix_p)
        q = matrix(vec_q)

        # subject to 
        matrix_a = []
        for i in range(n_samples):
            matrix_a.append([1.0])
            matrix_a.append([-1.0])

        vec_b = np.array([0.0])

        A = matrix(matrix_a)
        b = matrix(vec_b)
        
        # bounds for alpha (has to be positive & lower then C so size is 4 * n)
        matrix_g = np.vstack((np.eye(n_samples * 2), 
                              -np.eye(n_samples * 2)))
        vec_h = np.hstack((np.ones(n_samples * 2) * self.C, 
                           np.zeros(n_samples * 2)))
        
        G = matrix(matrix_g)
        h = matrix(vec_h)

        # set solver to silent call solver
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # extract support vectors and bias
        alpha = np.array(solution['x']).flatten()
        calculated_ys = solution['y'][0]
    
        self.alpha = np.array([alpha[::2], alpha[1::2]]).T

        self.support_vectors = X
        
        self.b = calculated_ys

        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predict the output for new data.
        """
        alpha_diffs = self._alpha_difference()

        preds = []
        for x in X:
            k_x = self.kernel(x, self.support_vectors)
            y_pred = np.dot(k_x, alpha_diffs) + self.b
            preds.append(y_pred)
        return np.array(preds)

    def get_alpha(self) -> np.array:
        """
        Return the alpha values.
        """
        return self.alpha

    def _alpha_difference(self) -> np.array:
        """
        Calculate the difference between alpha and alpha*.
        """
        alpha_diffs = []
        for [a1, a1_] in self.alpha:
            alpha_diffs.append(a1 - a1_)
        return np.array(alpha_diffs)

    def get_b(self) -> float:
        """
        Return the bias.
        """
        return self.b

if __name__ == "__main__":
    pass