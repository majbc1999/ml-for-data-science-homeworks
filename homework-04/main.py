import numpy as np
import pandas as pd
from typing import Union
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
        # Although that is not necessary as we have a reference class anyway
        u -= np.min(u)

        # Compute softmax
        exp_ = np.exp(u)
        sum_ = sum(np.exp(u))

        return exp_ / sum_

    def _neg_log_likelihood(self, coef: np.ndarray,
                                  X: np.ndarray, 
                                  y: np.ndarray,
                                  with_ref_class: bool = True) -> float:
        """
        Returns the log likelihood of the model.
        """
        coef = coef.reshape((self.n_classes - 1, X.shape[1]))
        # Append a vector of zeros to the beginning for the reference class
        if with_ref_class:
            coef = np.vstack([np.zeros((1, coef.shape[1])), coef])      

        neg_log_likelihood = 0
        for index, x in enumerate(X):
            probs = self._predict_single_sample(x, coef)
            true_class = y[index]
            if probs[y[index]] == 0:
                neg_log_likelihood -= 1000
            else:
                neg_log_likelihood -= np.log(probs[true_class])
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
        self.coef = np.zeros((self.n_classes - 1, X.shape[1]))
        # Optimization bounds
        bounds = [(None, None) for _ in range(self.coef.size)]

        # Run gradient descent
        new_coef = fmin_l_bfgs_b(self._neg_log_likelihood, 
                                 self.coef.flatten(), 
                                 args=(X, y), 
                                 bounds=bounds,
                                 approx_grad=True)[0]
        
        # Reshape and store coefficients
        self.coef = new_coef.reshape(self.n_classes - 1, X.shape[1])

        # Add a vector of zeros to the beginning for the reference class
        self.coef = np.vstack([np.zeros((1, X.shape[1])), self.coef])
        return self

    def predict(self, X):
        # Check if model is built
        if self.coef is None:
            raise Exception("Model not yet built")
        
        # Add a column of ones for the intercept
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        probs = []
        for x in X:
            probs.append(self._predict_single_sample(x, self.coef))
        return np.array(probs)

    def _predict_single_sample(self, x: np.array, coef: np.array) -> np.array:
        """
        Predicts a single sample.
        """
        # Compute u
        u = x @ coef.T

        # Return class with highest probability
        return self._softmax(u)

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
        coef = coef_and_deltas[:self.coef.size]
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

        for i in range(len(y)):
            if probabilities[i, y[i]] == 0:
                neg_log_likelihood -= 1000
            else:
                neg_log_likelihood -= np.log(probabilities[i, y[i]])
        return neg_log_likelihood

    def build(self, X: np.array, y: np.array):
        """
        Builds the model.
        """
        # Set number of classes
        self.n_classes = len(np.unique(y))

        if self.n_classes < 3:
            return ValueError("Invalid number of classes."
                             "Ordinal logistic regression only"
                             "makes sense for 3 or more classes.")

        # Add a column of ones for the intercept
        X = np.hstack([X, np.ones((len(X), 1))])

        # Initialize coefficients
        self.coef = np.zeros(len(X[0]))
        
        # Initialize deltas for thresholds
        self._deltas = np.ones(self.n_classes - 2)

        # Optimization bounds
        bounds = [(None, None) for _ in range(self.coef.size)] + \
                    [(0.01, None) for _ in range(self._deltas.size)]

        # Concatenate coefficients and deltas
        coef_and_deltas = np.concatenate((self.coef, self._deltas))

        print(coef_and_deltas)

        # Run gradient descent
        new_coef_and_deltas = fmin_l_bfgs_b(func=self._neg_log_likelihood,
                                            x0=coef_and_deltas,
                                            args=(X, y),
                                            bounds=bounds,
                                            approx_grad=True)[0]

        # Extract coefficients and deltas
        self.coef = new_coef_and_deltas[:self.coef.size]
        self._deltas = coef_and_deltas[self.coef.size:]

        # Check length of deltas
        if len(self._deltas) != self.n_classes - 2:
            raise Exception("Invalid number of deltas")

        # Convert deltas to thresholds
        self.t = self._deltas_to_thresholds(self._deltas)

        # Check length of thresholds
        if len(self.t) != self.n_classes + 1:
            raise Exception("Invalid number of thresholds")

        print(self.t)
        print(self.coef)

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
            raise ValueError("Model not yet built")
        
        else:
            probs = []
            u_i = coef.T @ x
            for i in range(self.n_classes):
                first = t[i + 1] - u_i if t[i + 1] != 'infty' else 'infty'
                second = t[i] - u_i if t[i] != '-infty' else '-infty'

                probs.append(
                    self._CDF(first) - self._CDF(second)
                )
            return np.array(probs)


# PART II: application

def draw_dependancy(df1: pd.Series, df2: pd.Series, names: tuple) -> None:
    """
    Draws a plot of dependancy between two series.
    """
    # Merge two series into one dataframe
    df = pd.concat([df1, df2], axis=1)
    df.columns = [names[0], names[1]]
    
    # Count proportion of x values for each y value
    df = df.groupby(names[0]).count()
    df = df / df.sum()
    df = df.reset_index()

    # Draw a plot
    plt.figure()
    plt.bar(df[names[0]], df[names[1]])
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.show()
    return

def data_preparation(path: str, numpy: bool = True, only_import: bool = False):
    """
    Function, where we prepare our data for further insight.
    """
    df = pd.read_csv(path, sep=';')

    # We have to convert discrete variables into numeric 
    # in order to use them in our model

    # 1. shot type to numeric
    def shot_type_to_num(x: str) -> int:
        """
        Converts shot type to numeric
        
        Maps are just so they fit our model
        """
        if x == 'above head':
            return 0
        elif x == 'layup':
            return 1
        elif x == 'other':
            return 2
        elif x == 'hook shot':
            return 3
        elif x == 'dunk':
            return 4
        elif x == 'tip-in':
            return 5
        else: 
            raise Exception("Invalid shot type")

    def competition_to_num(x: str) -> int:
        """
        Converts competition to numeric      
        """
        if x == 'U14':
            return 1
        elif x == 'U16':
            return 2
        elif x == 'NBA':
            return 7
        elif x == 'SLO1':
            return 4
        elif x == 'EURO':
            return 5
        else:
            raise Exception("Invalid competition type")

    def player_type_to_num(x: str) -> int:
        """Converts player type to numeric"""
        if x == 'G':
            return 0
        elif x == 'F':
            return 1
        elif x == 'C':
            return 5
        else:
            raise Exception("Invalid player type")

    # transition doesn't need to be converted as it has only 2 values
    # same for two_legged and made_shot

    def movement_to_num(x: str) -> int:
        """Converts movement to numeric"""
        if x == 'no':
            return 1
        elif x == 'dribble or cut':
            return 2
        elif x == 'drive':
            return 3
        else:
            raise Exception("Invalid movement type")

    # angle and distance can stay the same for now
    
    if not only_import:
        df['ShotType'] = df['ShotType'].map(shot_type_to_num)
        df['Competition'] = df['Competition'].map(competition_to_num)
        df['PlayerType'] = df['PlayerType'].map(player_type_to_num)
        df['Movement'] = df['Movement'].map(movement_to_num)

    y = df['ShotType']
    X = df.drop(['ShotType'], axis=1)
    
    cols = list(X.columns)

    # convert to numpy array
    if numpy:
        X = X.to_numpy()
        y = y.to_numpy()

    return X, y, cols

def multinomial_bad_ordinal_good():
    return


# PART III
MBOG_TRAIN = 100

if __name__ == '__main__':
    # here we go
    X, y, cols = data_preparation('homework-04/dataset.csv', numpy=False, 
                                  only_import = True)

    # draw dependancy between two columns
    for col in cols:
        #draw_dependancy(y, X[col], ('ShotType', col))
        pass

    X, y, cols = data_preparation('homework-04/dataset.csv')
