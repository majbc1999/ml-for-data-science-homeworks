import numpy as np
import cvxopt
from cvxopt import matrix
import pandas as pd
import matplotlib.pyplot as plt

# models

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
    def __init__(self, M: int = 2, gamma: float = 1.0):
        """
        Initialize the kernel with a M value.
        """
        self.M = M
        self.gamma = gamma

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
            kernel = (1 + self.gamma * np.dot(x, y)) ** self.M
            
        # 1D vector and 2D matrix
        elif x_dim == 1 and y_dim == 2:
            kernel = (1 + self.gamma * np.dot(x, y.T)) ** self.M
            kernel = kernel.squeeze()  # Remove singleton dimension
            
        # 2D matrix and 1D vector
        elif x_dim == 2 and y_dim == 1:
            kernel = (1 + self.gamma * np.dot(x, y)) ** self.M
            kernel = kernel.squeeze()  # Remove singleton dimension
            
        # Two 2D matrices
        elif x_dim == 2 and y_dim == 2:
            kernel = (1 + self.gamma * np.dot(x, y.T)) ** self.M
            
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
        self.X = None

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
        if self.X is None:
            raise ValueError("The model has not been fitted yet.")
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

    def produce_sparse_solution(self, tolerance: float):
        """
        Filter out the support vectors with alpha < tolerance.
        """
        support_vectors = []
        counter = 0
        
        new_alphas = []
        for i, [a1, a1_] in enumerate(self.alpha):
            if abs(a1 - a1_) > tolerance:
                support_vectors.append(self.support_vectors[i])
                new_alphas.append([a1, a1_])
            else:
                counter += 1
        print(f"Filtered out {counter} support vectors.")
        
        self.support_vectors = np.array(support_vectors)
        self.alpha = np.array(new_alphas)

        return self

    def get_support_vectors(self) -> np.array:
        """
        Return the support vectors.
        """
        return self.support_vectors


# helper functions

def calculate_mse(X_true: np.array, y_true: np.array, model) -> float:
    """
    Calculate the mean squared error.
    """
    y_pred = model.predict(X_true)
    return np.mean((y_true - y_pred) ** 2)

def sine_dataset_parameter_discovery(train, test):
    # Ia: kernelized ridge regression

    # 1. polynomial kernel (lambda doesn't matter as much as M)
    vals = []
    for M in range(1, 20):
        kernel = Polynomial(M)
        for lambda_ in [0.0001, 0.001, 0.01, 0.1, 1.0, 10]:
            model = KernelizedRidgeRegression(kernel, lambda_=lambda_)
            model.fit(train["x"].values.reshape(-1, 1), train["y"].values)
            mse = calculate_mse(test['x'].values.reshape(-1, 1), test['y'].values, model)
            vals.append([lambda_, M, mse])
    
    vals = np.array(vals)
    # normalize mse
    vals[:, 2] = np.log(vals[:, 2])

    plt.figure(figsize=(4, 4))
    plt.scatter(vals[:, 0], vals[:, 1], c=vals[:, 2], cmap='RdYlGn')
    plt.title('Kernelized Ridge Regression \n kernel = polynomial')
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('M')
    plt.savefig('homework-05/plots/ridge_poly.png', bbox_inches='tight')
    plt.show()

    # 2. RBF kernel
    vals = []
    for sigma in [0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]:
        kernel = RBF(sigma=sigma)
        for lambda_ in [0.0001, 0.001, 0.01, 0.1, 1.0, 10]:
            model = KernelizedRidgeRegression(kernel, lambda_=lambda_)
            model.fit(train["x"].values.reshape(-1, 1), train["y"].values)
            mse = calculate_mse(test['x'].values.reshape(-1, 1), test['y'].values, model)
            vals.append([lambda_, sigma, mse])
    
    vals = np.array(vals)
    # normalize mse
    vals[:, 2] = np.log(vals[:, 2])

    plt.figure(figsize=(4, 4))
    plt.scatter(vals[:, 0], vals[:, 1], c=vals[:, 2], cmap='RdYlGn')
    plt.title('Kernelized Ridge Regression \n kernel = RBF')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('sigma')
    plt.savefig('homework-05/plots/ridge_rbf.png', bbox_inches='tight')
    plt.show()

    # Ib: support vector regression
    # 1. polynomial kernel
    vals = []
    for M in range(1,5):
        kernel = Polynomial(M=M)
        for epsilon in [0.0001, 0.001, 0.01, 0.1, 1.0]:
            model = SVR(kernel, lambda_=0.01, epsilon=epsilon)
            model.fit(train["x"].values.reshape(-1, 1), train["y"].values)
            mse = calculate_mse(test['x'].values.reshape(-1, 1), test['y'].values, model)
            vals.append([epsilon, M, mse])
    
    vals = np.array(vals)
    # normalize mse
    vals[:, 2] = np.log(vals[:, 2])

    plt.figure(figsize=(4, 4))
    plt.scatter(vals[:, 0], vals[:, 1], c=vals[:, 2], cmap='RdYlGn')
    plt.title('Support Vector Regression \n kernel = Polynomial')
    plt.xscale('log')
    plt.xlabel('epsilon')
    plt.ylabel('M')
    plt.savefig('homework-05/plots/svr_poly.png', bbox_inches='tight')
    plt.show()

    # 2. RBF kernel
    vals = []
    for sigma in [0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]:
        kernel = RBF(sigma=sigma)
        for epsilon in [0.0001, 0.001, 0.01, 0.1, 1.0]:
            model = SVR(kernel, lambda_=0.01, epsilon=epsilon)
            model.fit(train["x"].values.reshape(-1, 1), train["y"].values)
            mse = calculate_mse(test['x'].values.reshape(-1, 1), test['y'].values, model)
            vals.append([epsilon, sigma, mse])
    
    vals = np.array(vals)
    # normalize mse
    vals[:, 2] = np.log(vals[:, 2])

    plt.figure(figsize=(4, 4))
    plt.scatter(vals[:, 0], vals[:, 1], c=vals[:, 2], cmap='RdYlGn')
    plt.title('Support Vector Regression \n kernel = RBF')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('epsilon')
    plt.ylabel('sigma')
    plt.savefig('homework-05/plots/svr_rbf.png', bbox_inches='tight')
    plt.show()

def cross_validate_mse(kernel, model, X, y, lambda_, k=5):
    if model == "SVR":
        # 1st k-fold
        model = SVR(kernel=kernel, lambda_=lambda_)
        crs = len(X) // k
        X_train = X[crs:]
        y_train = y[crs:]
        X_test = X[:crs]
        y_test = y[:crs]
        model.fit(X_train, y_train)

        mses = []
        mses.append(calculate_mse(X_test, y_test, model))

        for i in range(k - 2):
            X_train = np.concatenate((X[:crs * i], X[crs * (i + 1):]))
            y_train = np.concatenate((y[:crs * i], y[crs * (i + 1):]))
            X_test = X[crs * i:crs * (i + 1)]
            y_test = y[crs * i:crs * (i + 1)]
            model = SVR(kernel=kernel, lambda_=lambda_)
            model.fit(X_train, y_train)
            mses.append(calculate_mse(X_test, y_test, model))

        # last k-fold
        X_train = X[:crs * (k - 1)]
        y_train = y[:crs * (k - 1)]
        X_test = X[crs * (k - 1):]
        y_test = y[crs * (k - 1):]
        model = SVR(kernel=kernel, lambda_=lambda_)
        model.fit(X_train, y_train)
        mses.append(calculate_mse(X_test, y_test, model))

        return np.mean(mses)

    else:
        # 1st k-fold
        model = KernelizedRidgeRegression(kernel=kernel, lambda_=lambda_)
        crs = len(X) // k
        X_train = X[crs:]
        y_train = y[crs:]
        X_test = X[:crs]
        y_test = y[:crs]
        model.fit(X_train, y_train)

        mses = []
        mses.append(calculate_mse(X_test, y_test, model))

        for i in range(k - 2):
            X_train = np.concatenate((X[:crs * i], X[crs * (i + 1):]))
            y_train = np.concatenate((y[:crs * i], y[crs * (i + 1):]))
            X_test = X[crs * i:crs * (i + 1)]
            y_test = y[crs * i:crs * (i + 1)]
            model = KernelizedRidgeRegression(kernel=kernel, lambda_=lambda_)
            model.fit(X_train, y_train)
            mses.append(calculate_mse(X_test, y_test, model))

        # last k-fold
        X_train = X[:crs * (k - 1)]
        y_train = y[:crs * (k - 1)]
        X_test = X[crs * (k - 1):]
        y_test = y[crs * (k - 1):]
        model = KernelizedRidgeRegression(kernel=kernel, lambda_=lambda_)
        model.fit(X_train, y_train)
        mses.append(calculate_mse(X_test, y_test, model))

        return np.mean(mses)

def optimal_lambda(kernel, model, X, y, k=5):
    lambdas1 = np.linspace(0.001, 1, 20)
    lambdas2 = np.linspace(1, 100, 20)
    lambdas = np.concatenate((lambdas1, lambdas2))
    mses = []
    for lambda_ in lambdas:
        mses.append(cross_validate_mse(kernel, model, X, y, lambda_, k=k))
    return lambdas[np.argmin(mses)]

# script functions

def sine_SVR_polynomial(X, y):
    """
    Sine dataset with polynomial kernel and SVR
    """
    kernel1 = Polynomial(9, 0.05)
    model1 = SVR(kernel=kernel1, lambda_=0.1, epsilon=1.2)
    model1 = model1.fit(X, y)
    model1 = model1.produce_sparse_solution(tolerance=1e-3)

    support_vectors = model1.support_vectors
    print(f'Length of support vectors: {len(support_vectors)}')

    support_ys = []
    for vector in support_vectors:
        # find corresponding y value
        index = np.where((X == vector).all(axis=1))[0][0]
        support_ys.append(y[index])

    fit_X = np.linspace(0, 20, 1000)
    fit_X = fit_X.reshape(-1, 1)

    fit_y = model1.predict(fit_X)
    fit_X = fit_X.flatten()

    # plot the input data, the fit, and mark support vectors on the plot
    plt.figure()
    plt.scatter(X.flatten(), y, color="tab:orange", label="data")
    plt.plot(fit_X, fit_y, color="tab:blue", label="fit")
    plt.scatter(support_vectors, support_ys, color="tab:red", label="support vectors", edgecolors="black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("SVR with polynomial kernel (M = 9, lambda = 0.1, epsilon = 1.2)")
    plt.legend()
    plt.savefig("homework-05/plots/sine_SVR_polynomial.png")

def sine_SVR_RBF(X, y):
    """
    Sine dataset with RBF kernel and SVR
    """
    kernel2 = RBF(1.8)

    model1 = SVR(kernel=kernel2, lambda_=0.1, epsilon=1)
    model1 = model1.fit(X, y)
    model1 = model1.produce_sparse_solution(tolerance=1e-3)

    support_vectors = model1.support_vectors
    print(f'Length of support vectors: {len(support_vectors)}')

    support_ys = []
    for vector in support_vectors:
        # find corresponding y value
        index = np.where((X == vector).all(axis=1))[0][0]
        support_ys.append(y[index])

    fit_X = np.linspace(0, 20, 1000)
    fit_X = fit_X.reshape(-1, 1)

    fit_y = model1.predict(fit_X)
    fit_X = fit_X.flatten()

    # plot the input data, the fit, and mark support vectors on the plot
    plt.figure()
    plt.scatter(X.flatten(), y, color="tab:orange", label="data")
    plt.plot(fit_X, fit_y, color="tab:blue", label="fit")
    plt.scatter(support_vectors, support_ys, color="tab:red", label="support vectors", edgecolors="black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("SVR with RBF kernel (sigma = 1.8, lambda = 0.1, epsilon = 1)")
    plt.legend()
    plt.savefig("homework-05/plots/sine_SVR_RBF.png")

def sine_KRR_polynomial(X, y):
    """
    Sine dataset with polynomial kernel and kernelized ridge regression
    """
    kernel2 = Polynomial(9, 0.05)

    model1 = KernelizedRidgeRegression(kernel=kernel2, lambda_=0.1)
    model1 = model1.fit(X, y)

    fit_X = np.linspace(0, 20, 1000)
    fit_X = fit_X.reshape(-1, 1)

    fit_y = model1.predict(fit_X)
    fit_X = fit_X.flatten()

    # plot the input data, the fit, and mark support vectors on the plot
    plt.figure()
    plt.scatter(X.flatten(), y, color="tab:orange", label="data")
    plt.plot(fit_X, fit_y, color="tab:blue", label="fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("KRR with polynomial kernel (M = 9, lambda = 0.1)")
    plt.legend()
    plt.savefig("homework-05/plots/sine_KRR_polynomial.png")

def sine_KRR_RBF(X, y):
    """
    Sine dataset with RBF kernel and kernelized ridge regression
    """
    kernel2 = RBF(1.8)

    model1 = KernelizedRidgeRegression(kernel=kernel2, lambda_=0.1)
    model1 = model1.fit(X, y)

    fit_X = np.linspace(0, 20, 1000)
    fit_X = fit_X.reshape(-1, 1)

    fit_y = model1.predict(fit_X)
    fit_X = fit_X.flatten()

    # plot the input data, the fit, and mark support vectors on the plot
    plt.figure()
    plt.scatter(X.flatten(), y, color="tab:orange", label="data")
    plt.plot(fit_X, fit_y, color="tab:blue", label="fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("KRR with RBF kernel (sigma = 1.8, lambda = 0.1)")
    plt.legend()
    plt.savefig("homework-05/plots/sine_KRR_RBF.png")

def krr_and_polynomial_housing(test_X, test_y, train_X, train_y):

    mses = []

    for m in range(1, 11):
        
        model = KernelizedRidgeRegression(kernel=Polynomial(m, 0.05), 
                                          lambda_=1)
        model = model.fit(train_X, train_y)
        mse = calculate_mse(test_X, test_y, model)
        
        opt_lambda = optimal_lambda(Polynomial(m, 0.05), "KRR", train_X, train_y)
        model2 = KernelizedRidgeRegression(kernel=Polynomial(m, 0.05),
                                           lambda_=opt_lambda)
        model2 = model2.fit(train_X, train_y)
        mse2 = calculate_mse(test_X, test_y, model2)

        mses.append([m, mse, mse2])

    mses = np.array(mses)
    print(mses)

    # plot
    plt.figure()
    plt.plot(mses[:, 0], mses[:, 1], label="lambda = 1", color="red")
    plt.plot(mses[:, 0], mses[:, 2], label="lambda = optimal", color="blue")
    plt.xlabel("M")
    plt.ylabel("MSE")
    plt.title("MSE for M = 1, 2, ..., 10 for polynomial kernel \n (Kernelized Ridge Regression)")
    plt.legend()
    plt.savefig("homework-05/plots/mse_comparison1.png")
    plt.show()

def svr_and_polynomial_housing(test_X, test_y, train_X, train_y):
    mses = []

    for m in range(1, 11):
        
        model = SVR(kernel=Polynomial(m), 
                           lambda_=1)
        model = model.fit(train_X, train_y)
        mse = calculate_mse(test_X, test_y, model)
        
        opt_lambda = optimal_lambda(Polynomial(m, 1.4), "SVR", train_X, train_y)
        print(opt_lambda)
        model2 = SVR(kernel=Polynomial(m),
                     lambda_=opt_lambda)
        model2 = model2.fit(train_X, train_y)
        mse2 = calculate_mse(test_X, test_y, model2)

        mses.append([m, mse, mse2])

    mses = np.array(mses)
    print(mses)

    # plot
    plt.figure()
    plt.plot(mses[:, 0], mses[:, 1], label="lambda = 1", color="red")
    plt.plot(mses[:, 0], mses[:, 2], label="lambda = optimal", color="blue")
    plt.xlabel("M")
    plt.ylabel("MSE")
    plt.title("MSE for M = 1, 2, ..., 10 for polynomial kernel \n (SVR)")
    plt.legend()
    plt.savefig("homework-05/plots/mse_comparison2.png")
    plt.show()

def krr_and_rbf_housing(test_X, test_y, train_X, train_y):
    mses = []

    for sigma in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        
        model = KernelizedRidgeRegression(kernel=RBF(sigma), 
                           lambda_=1)
        model = model.fit(train_X, train_y)
        mse = calculate_mse(test_X, test_y, model)
        
        opt_lambda = optimal_lambda(RBF(sigma), "SVR", train_X, train_y)
        print(opt_lambda)
        model2 = KernelizedRidgeRegression(kernel=RBF(sigma),
                     lambda_=opt_lambda)
        model2 = model2.fit(train_X, train_y)
        mse2 = calculate_mse(test_X, test_y, model2)

        mses.append([sigma, mse, mse2])

    mses = np.array(mses)
    print(mses)

    # plot
    plt.figure()
    plt.plot(mses[:, 0], mses[:, 1], label="lambda = 1", color="red")
    plt.plot(mses[:, 0], mses[:, 2], label="lambda = optimal", color="blue")
    plt.xlabel("M")
    plt.ylabel("MSE")
    plt.title("MSE for sigma = 0.001, ..., 1000 for rbf kernel \n (Kernelized Ridge Regression)")
    plt.legend() 
    plt.savefig("homework-05/plots/mse_comparison3.png")
    plt.show()

def svr_and_rbf_housing(test_X, test_y, train_X, train_y):
    mses = []

    for sigma in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        
        model = SVR(kernel=RBF(sigma), 
                    lambda_=1)
        model = model.fit(train_X, train_y)
        mse = calculate_mse(test_X, test_y, model)
        
        opt_lambda = optimal_lambda(RBF(sigma), "SVR", train_X, train_y)
        print(opt_lambda)
        model2 = SVR(kernel=RBF(sigma),
                     lambda_=opt_lambda)
        model2 = model2.fit(train_X, train_y)
        mse2 = calculate_mse(test_X, test_y, model2)

        mses.append([sigma, mse, mse2])

    mses = np.array(mses)
    print(mses)

    # plot
    plt.figure()
    plt.plot(mses[:, 0], mses[:, 1], label="lambda = 1", color="red")
    plt.plot(mses[:, 0], mses[:, 2], label="lambda = optimal", color="blue")
    plt.xlabel("M")
    plt.ylabel("MSE")
    plt.title("MSE for sigma = 0.001, ..., 1000 for rbf kernel \n (SVR)")
    plt.legend()
    plt.savefig("homework-05/plots/mse_comparison4.png")
    plt.show()

if __name__ == "__main__":
#    # -------------------------------------------------------------------------
#    # I: sine dataset part (finished)
#
#    # load the data
#    df = pd.read_csv("homework-05/sine.csv")
#    
#    # randomly select 80% of the data for training
#    train = df.sample(frac=0.8, random_state=1)
#    # the other 20% is for testing
#    test = df.drop(train.index)
#    
#    # split the data into X and y
#    X = df.drop('y', axis=1).to_numpy()
#    y = df['y'].to_numpy()
#
#    # plot the input data, the fit and support vectors for each model
#    sine_SVR_polynomial(X, y)
#    sine_SVR_RBF(X, y)
#    sine_KRR_polynomial(X, y)
#    sine_KRR_RBF(X, y)

    # -------------------------------------------------------------------------
    # II. housing dataset part
    # load the data
    df = pd.read_csv("homework-05/housing2r.csv")

    y = df["y"]
    X = df.drop("y", axis=1)

    # first 80% of the data for training
    n_rows = int(df.shape[0] * 0.8)

    # select the first 80% of rows using iloc
    train_y = y.iloc[:n_rows].to_numpy()
    train_X = X.iloc[:n_rows, :].to_numpy()

    # select the remaining 20% of rows using iloc
    test_y = y.iloc[n_rows:].to_numpy()
    test_X = X.iloc[n_rows:, :].to_numpy()

    # perform the experiments
    krr_and_polynomial_housing(test_X, test_y, train_X, train_y)
      
    # svr_and_polynomial_housing(test_X, test_y, train_X, train_y)
    # 
    # krr_and_rbf_housing(test_X, test_y, train_X, train_y)
    # 
    # svr_and_rbf_housing(test_X, test_y, train_X, train_y)