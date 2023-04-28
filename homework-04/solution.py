import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import random

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
        u -= np.max(u)

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
        if with_ref_class:
            coef = coef.reshape((self.n_classes - 1, X.shape[1]))
            # Append a vector of zeros to the beginning for the reference class
            coef = np.vstack([np.zeros((1, coef.shape[1])), coef])
        else:
            coef = coef.reshape((self.n_classes, X.shape[1]))

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

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns the log likelihood of the model.
        """
        # Check if model is built
        if self.coef is None:
            raise Exception("Model not yet built")

        # Add a column of ones for the intercept
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        return - self._neg_log_likelihood(self.coef.flatten(), X, y, 
                                          with_ref_class=False)

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
        self._deltas = None  # threshold deltas

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

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns the log likelihood of the model.
        """
        # Check if model is built
        if self.coef is None:
            raise Exception("Model not yet built")

        # Add a column of ones for the intercept
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        # Build deltas and coefs
        coef_and_deltas = np.concatenate((self.coef, self._deltas))

        return -self._neg_log_likelihood(coef_and_deltas, X, y)


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
        Converts competition to numeric.

        U14 and U16 are not so different as they have players with only 2 years
        difference, so we can map them to similar numbers.

        SLO1 and EURO are also similar, as they are both european leagues.

        NBA is the most different, as it is the best league in the world, with 
        slightly different rules, so we map it to the highest number.   
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
        """
        Converts player type to numeric
        
        Center is the most different player, as he almost never shoots from 
        distance and always plays under the basket, so we map him to the lowest
        number.

        Guard and forward are similar, as they both play on the perimeter and
        shoot from distance, so we map them to similar numbers.
        """
        if x == 'G':
            return 4
        elif x == 'F':
            return 3
        elif x == 'C':
            return 1
        else:
            raise Exception("Invalid player type")

    # transition doesn't need to be converted as it has only 2 values, it will
    # anyway be normalized later

    # same for two_legged and made_shot

    def movement_to_num(x: str) -> int:
        """
        Converts movement to numeric.

        No movement is the most different so it is mapped to 1. Dribble or 
        cut and drive both resemble movement, so they are mapped to similar
        numbers.
        """
        if x == 'no':
            return 1
        elif x == 'dribble or cut':
            return 3
        elif x == 'drive':
            return 4
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

def normalize_data(X: np.array):
    """
    Normalizes the given dataset by subtracting the mean and dividing by the standard deviation
    :param X: The dataset to be normalized
    :return: The normalized dataset
    """
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_norm

def resample(X: np.array, y: np.array, n_samples: int) -> tuple:
    """
    Samples n_samples from the given dataset X and y with replacement
    """
    # sample indices
    indices = np.random.choice(len(X), n_samples, replace=True)

    # sample data
    X_sample = X[indices]
    y_sample = y[indices]

    # add one made up instance of each class, to make sure we have all classes
    for i in np.unique(y):
        if i not in y_sample:
            X_sample = np.vstack((X_sample, np.ones(len(X[0]))))
            y_sample = np.append(y_sample, i)

    return X_sample, y_sample

def bootstrap_confidence_interval(X: np.array, y: np.array,
                                  n: int = 100, m: int = 50) -> list:
    """
    Calculates the confidence interval for coefficients the given model 
    using the bootstrap method. We build the model `n` times on `m` randomly 
    sampled data.
    """
    coef = []

    # build model n times
    for i in range(n):
        model = MultinomialLogReg()
        # sample data
        X_sample, y_sample = resample(X, y, n_samples=m)

        # build model
        model = model.build(X_sample, y_sample)
        print(f'Finished: {i+1}/{n}', end='\r')

        # append coefficients
        coef.append(model.coef)
    print(f'Finished: {n}/{n}')

    # calculate confidence interval for each coefficient
    confidence = []

    for i in range(len(coef[0])):
        row = []
        for j in range(len(coef[0][0])):
            # get all coefficients for i-th feature
            coef_i = [coef[k][i][j] for k in range(n)]
            # calculate 5% confidence interval
            row.append(np.percentile(coef_i, [2.5, 97.5]))
        confidence.append(row)

    return confidence

def multinomial_bad_ordinal_good(n_samples: int = 100, rand: random.Random = None) -> tuple:
    """
    A data generating process of data, where ordinal logistic regression
    performs beter than multinomial logistic regression.
    
    Dataset will have two attributes:
        - last game points of this team (0, 1 or 2)
        - last game points of oposing team (0, 1 or 2)
        - strength of the team (0, 1)
        - strength of the oposing team (0, 1)

    Target variable will have 3 possible values, winning (2), losing (0) 
    and drawing (1).
    """

    # Probabilities based on point difference
    def return_probs(a: int, strength1: int, strength2: int) -> list:
        if a <= -2:
            p = [0.1, 0.1, 0.8]
        elif a == -1:
            p = [0.2, 0.3, 0.5]
        elif a == 0:
            p = [0.3, 0.4, 0.3]
        elif a == 1:
            p = [0.5, 0.3, 0.2]
        elif a >= 2:
            p = [0.8, 0.1, 0.1]
        else:
            return ValueError("Invalid point difference")
        
        if strength1 == strength2:
            return p
        elif strength1 > strength2:
            return [p[0] + 0.1, p[1] - 0.05, p[2] - 0.05]
        else:
            return [p[0] - 0.05, p[1] - 0.05, p[2] + 0.1]

    X = []
    y = []

    for _ in range(n_samples):
        # randomly generate X
        x1 = rand.choice([0, 1, 3])
        x2 = rand.choice([0, 1, 3])
        x3 = rand.choice([0, 1])
        x4 = rand.choice([0, 1])

        probs_for_y = return_probs(x1 - x2, x3, x4)

        # sample with probabilities
        X.append([x1, x2, x3, x4])
        y.append(rand.choices([0, 1, 2], weights=probs_for_y)[0])

    return np.array(X), np.array(y)


# PART III
MBOG_TRAIN = 1000

if __name__ == '__main__':
    # here we go
    X, y, cols = data_preparation('homework-04/dataset.csv')

    # normalize dataset
    X = normalize_data(X)

    # build model
    model = MultinomialLogReg().build(X[-50:], y[-50:])

    # map shot types to names
    shots = ['above head (reference class)', 'layup',
             'other', 'hook shot', 'dunk', 'tip-in']

    model.coef = np.round(model.coef, decimals=3)

    # make pandas dataframe
    df = pd.DataFrame(model.coef)
    df['shot type'] = shots
    df = df.rename(columns={0: 'Competition',
                            1: 'PlayerType',
                            2: 'Transition',
                            3: 'TwoLegged',
                            4: 'Movement',
                            5: 'Angle',
                            6: 'Distance',
                            7: 'Intercept'})

    # Put shot type to the beginning
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    # set column shot type as index
    df = df.set_index('shot type')

    # print coefficients
    print('Coefficients:')
    print(df)

    # calculate confidence interval
    confidence = bootstrap_confidence_interval(X, y, n=5, m=10)

    for i in range(len(confidence)):
        for j in range(len(confidence[0])):
            confidence[i][j] = np.round(confidence[i][j], decimals=2)
            confidence[i][j] = (confidence[i][j][0], confidence[i][j][1])

    # change confidence interval to pandas dataframe, same as above
    df2 = pd.DataFrame(confidence)

    shots = ['above head (reference class)', 'layup',
             'other', 'hook shot', 'dunk', 'tip-in']

    df2['shot type'] = shots
    df2 = df2.rename(columns={0: 'Competition',
                            1: 'PlayerType',
                            2: 'Transition',
                            3: 'TwoLegged',
                            4: 'Movement',
                            5: 'Angle',
                            6: 'Distance',
                            7: 'Intercept'})

    # Put shot type to the beginning
    cols = df2.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df2 = df2[cols]
    # set column shot type as index
    df2 = df2.set_index('shot type')

    # print confidence interval
    print('Confidence interval:')
    print(df2)
    
    # generate data
    X, y = multinomial_bad_ordinal_good(n_samples=MBOG_TRAIN+1000,
                                        rand=random.Random(x=42))

    # split data
    X_train, X_test = X[:MBOG_TRAIN], X[MBOG_TRAIN:]
    y_train, y_test = y[:MBOG_TRAIN], y[MBOG_TRAIN:]

    # build models
    model_ordinal = OrdinalLogReg().build(X_train, y_train)
    model_multinomial = MultinomialLogReg().build(X_train, y_train)

    # compare log-likelihoods
    print('Log-likelihoods:')
    print(f'    Multinomial: \t'
          f'{model_multinomial.log_likelihood(X_test, y_test)}')
    print(f'    Ordinal: \t \t'
          f'{model_ordinal.log_likelihood(X_test, y_test)}')

