import numpy as np
import pandas as pd
from typing import Tuple


def toy_data(n: int, seed: int = None) -> pd.DataFrame:
    """
    Generates n examples of Toy data for our dataset.

    First 5 columns (attributes) of a dataset `X` contribute towards
    `y`, and the last 3 don't.
    """
    np.random.seed(seed)
    x = np.random.normal(size=(n, 8))
    z = 0.4 * x[:, 0] - 0.5 * x[:, 1] + 1.75 * \
        x[:, 2] - 0.2 * x[:, 3] + x[:, 4]
    y = np.random.uniform(size=n) > 1 / (1 + np.exp(-z))
    return pd.DataFrame({'x': x.tolist(), 'y': y})


def log_loss(y: float, p: float) -> float:
    """
    Calculates log loss for given y and p
    """
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def split_dataset(dataset: pd.DataFrame,
                  seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into two datasets of equal size
    """
    length = len(dataset)
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    dataset1 = dataset.iloc[indices[:length//2]]
    dataset2 = dataset.iloc[indices[length//2:]]
    return dataset1, dataset2


def model_risks(y: np.array, probabilities: np.array) -> np.array:
    """
    Returns the model risks for each example in the dataset
    """
    model_risks = []
    for i in range(len(y)):
        a = 1 if y[i] else 0
        model_risks.append(log_loss(a, probabilities[i][1]))
    return model_risks


def standard_error_of_model_risk(risks: np.array) -> float:
    """
    Returns the standard error of the model risk estimate
    """
    return np.std(risks, ddof=1) / np.sqrt(len(risks))


def confidence_interval_of_model_risk(risks: np.array) -> Tuple[float, float]:
    """
    Returns the 95% confidence interval of the model risk
    """
    std_error = standard_error_of_model_risk(risks)
    mean_risk = np.mean(risks)
    return (mean_risk - 1.96 * std_error, mean_risk + 1.96 * std_error)


def calculate_baseline_true_risk(true_risks: np.array,
                                 true_risk_proxy: float) -> float:
    """
    Compute 0.5 - 0.5 risk
    """
    num_risks_below_proxy = sum(risk < true_risk_proxy for risk in true_risks)
    num_risks_above_proxy = sum(risk >= true_risk_proxy for risk in true_risks)
    if num_risks_below_proxy >= num_risks_above_proxy:
        return (num_risks_below_proxy / len(true_risks))
    else:
        return (num_risks_above_proxy / len(true_risks))


def median_standard_error(estimates: np.array) -> float:
    """
    Calculates the median standard error for a list of estimates.
    """
    # Calculate the standard error of the mean for each estimate
    standard_errors = np.std(estimates, ddof=1) / np.sqrt(len(estimates))

    # Calculate the median of the standard errors
    median_standard_error = np.median(standard_errors)

    return median_standard_error


def is_in_ci(risks: np.array) -> float:
    """
    Computes the % of times the true risk is in the 95% confidence interval.
    """
    contains_risk = 0
    for (_, in_ci) in risks:
        if in_ci:
            contains_risk += 1
    return 100 * contains_risk / len(risks)


def k_fold_cross_validation(X: np.array,
                            y: np.array,
                            k: int,
                            model: object,
                            true_risk_proxy: float,
                            random_seed: int) -> Tuple[float, bool]:
    """
    Performs k-fold cross validation on the data X, y and returns the average
    difference between the true risk and the estimated risk and whether or not
    confidence interval contains the true risk proxy.
    """
    # Set the random seed
    np.random.seed(random_seed)

    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split the data into k folds
    step = len(X) // k
    risk_estimates = []

    # For each fold, train the model on the remaining data and compute the risk
    for i in range(k):

        # Split the data into training and test sets
        X_train = np.vstack((X[:i*step], X[(i+1)*step:]))
        y_train = np.hstack((y[:i*step], y[(i+1)*step:]))

        X_test = X[i*step:(i+1)*step]
        y_test = y[i*step:(i+1)*step]

        # Train the model and compute the risk
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)
        risk_estimates += model_risks(y_test, probabilities)

    risk_estimate_diff = np.mean(risk_estimates) - true_risk_proxy

    # Test if 95% confidence interval of the risk estimate contains the true risk
    ci = confidence_interval_of_model_risk(risk_estimates)
    if true_risk_proxy > ci[0] and true_risk_proxy < ci[1]:
        return (risk_estimate_diff, True)
    else:
        return (risk_estimate_diff, False)


def leave_one_out_cross_validation(X: np.array,
                                   y: np.array,
                                   model: object,
                                   true_risk_proxy: float,
                                   random_seed: int) -> Tuple[float, bool]:
    """
    Performs leave-one-out cross validation on the data X, y and returns the
    average difference between the true risk and the estimated risk and whether
    or not confidence interval contains the true risk proxy.
    """
    # Set the random seed
    np.random.seed(random_seed)

    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split the data into k folds
    risk_estimates = []

    # For each fold, train the model on the remaining data and compute the risk
    for i in range(len(X)):

        # Split the data into training and test sets
        X_train = np.vstack((X[:i], X[i+1:]))
        y_train = np.hstack((y[:i], y[i+1:]))

        X_test = X[i]
        y_test = y[i]

        # Train the model and compute the risk
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test.reshape(1, -1))
        risk_estimates += model_risks(np.array([y_test]), probabilities)

    risk_estimate_diff = np.mean(risk_estimates) - true_risk_proxy

    # Test if 95% confidence interval of the risk estimate contains the true risk
    ci = confidence_interval_of_model_risk(risk_estimates)
    if true_risk_proxy > ci[0] and true_risk_proxy < ci[1]:
        return (risk_estimate_diff, True)
    else:
        return (risk_estimate_diff, False)


def n_times_k_cross_validation(n: int,
                               X: np.array,
                               y: np.array,
                               k: int,
                               model: object,
                               true_risk_proxy: float,
                               random_seed: int) -> Tuple[float, bool]:
    """
    Repeat 10-fold cross-validation for 20 different partitions and let 
    each observation’s loss be the average of these 10.
    """
    # Set the random seed
    np.random.seed(random_seed)

    all_risk_estimates = [[] for _ in range(len(X))]

    for j in range(n):
        # Shuffle the data
        indices = np.random.permutation(len(X))

        X = X[indices]
        y = y[indices]

        # Return average risk estimate for each observation
        risk_estimates = []

        step = len(X) // k

        for i in range(k):
            # Split the data into training and test sets
            X_train = np.vstack((X[:i*step], X[(i+1)*step:]))
            y_train = np.hstack((y[:i*step], y[(i+1)*step:]))

            X_test = X[i*step:(i+1)*step]
            y_test = y[i*step:(i+1)*step]

            # Train the model and compute the risk
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_test)

            # Return risks for each observation
            risks = model_risks(y_test, probabilities)

            # Add the risks to the risk estimates
            for r in risks:
                risk_estimates.append(r)

        # Add the risk estimates to the list of all risk estimates
        for index, perm_index in enumerate(indices):
            all_risk_estimates[perm_index].append(risk_estimates[index])

    all_risk_estimates = [np.mean(risk) for risk in all_risk_estimates]
    risk_estimate_diff = np.mean(all_risk_estimates) - true_risk_proxy

    # Test if 95% confidence interval of the risk estimate contains the true risk
    ci = confidence_interval_of_model_risk(all_risk_estimates)
    if true_risk_proxy > ci[0] and true_risk_proxy < ci[1]:
        return (risk_estimate_diff, True)
    else:
        return (risk_estimate_diff, False)
