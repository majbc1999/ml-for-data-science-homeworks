from data_gen import toy_data
from loss_variability_due_to_data import model_risks, confidence_interval_of_model_risk, median_standard_error
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def k_fold_cross_validation(X, y, k, model, true_risk_proxy, random_seed=1000):
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
    
    # Test if 95% confidence interval of the risk estimate contains the true risk
    ci = confidence_interval_of_model_risk(risk_estimates)
    if true_risk_proxy > ci[0] and true_risk_proxy < ci[1]:
        return ([np.mean(risk_estimates)], True)
    else:
        return ([np.mean(risk_estimates)], False)


def leave_one_out_cross_validation(X, y, model, true_risk_proxy, random_seed=1000):
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
    
    # Test if 95% confidence interval of the risk estimate contains the true risk
    ci = confidence_interval_of_model_risk(risk_estimates)
    if true_risk_proxy > ci[0] and true_risk_proxy < ci[1]:
        return ([np.mean(risk_estimates)], True)
    else:
        return ([np.mean(risk_estimates)], False)


def n_times_k_cross_validation(n, X, y, k, 
                               model, true_risk_proxy, random_seed=1000):
    """
    Performs n times k-fold cross validation on the data X, y and returns the
    average difference between the true risk and the estimated risk and whether
    or not confidence interval contains the true risk proxy.
    """
    # Set the random seed
    np.random.seed(random_seed)

    # Shuffle the data
    for _ in range(n):
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

    # Test if 95% confidence interval of the risk estimate contains the true risk
    ci = confidence_interval_of_model_risk(risk_estimates)
    if true_risk_proxy > ci[0] and true_risk_proxy < ci[1]:
        return ([np.mean(risk_estimates)], True)
    else:
        return ([np.mean(risk_estimates)], False)


# 1. generate a toy dataset with 100 samples
toy_dataset = toy_data(100, seed=1002)


# 2. train model h_0 and compute true risk proxy
huge_dataset = toy_data(100000, seed=1001)

X = np.vstack(toy_dataset['x'].to_numpy())
y = np.array(toy_dataset['y'])

h_0 = LogisticRegression().fit(X, y)

huge_X = np.vstack(huge_dataset['x'].to_numpy())
huge_y = np.array(huge_dataset['y'])
huge_probabilities = h_0.predict_proba(huge_X)

true_risks = model_risks(huge_y, huge_probabilities)
true_risk_proxy = np.mean(true_risks)

# 3. Estimate h_0's risk using 5 estimators and repeat 500 times:

# a) 2-fold cross validation
# b) leave-one-out cross validation
# c) 10-fold cross validation
# d) 4-fold cross validation
# e) 10-fold cross validation repeated 20 times

saving_dict = {
    '2-fold': [[],[]],
    'leave-one-out': [[],[]],
    '10-fold': [[],[]],
    '4-fold': [[],[]],
    '20-10-fold': [[],[]]
}

name_key = {
    '2-fold': '2-fold cross validation',
    'leave-one-out': 'leave-one-out cross validation',
    '10-fold': '10-fold cross validation',
    '4-fold': '4-fold cross validation',
    '20-10-fold': '10-fold cross validation repeated 20 times'
}

for i in range(50):
    # a) 2-fold cross validation
    risks_list, ci = k_fold_cross_validation(X, y, 2, h_0, true_risk_proxy, random_seed=2000+i)
    saving_dict['2-fold'][0] += risks_list
    saving_dict['2-fold'][1].append(ci)
    # b) leave-one-out cross validation
    risks_list, ci = leave_one_out_cross_validation(X, y, h_0, true_risk_proxy, random_seed=2000+i)
    saving_dict['leave-one-out'][0] += risks_list
    saving_dict['leave-one-out'][1].append(ci)
    # c) 10-fold cross validation
    risks_list, ci = k_fold_cross_validation(X, y, 10, h_0, true_risk_proxy, random_seed=2000+i)
    saving_dict['10-fold'][0] += risks_list
    saving_dict['10-fold'][1].append(ci)
    # d) 4-fold cross validation
    risks_list, ci = k_fold_cross_validation(X, y, 4, h_0, true_risk_proxy, random_seed=2000+i)
    saving_dict['4-fold'][0] += risks_list
    saving_dict['4-fold'][1].append(ci)
    # e) 10-fold cross validation repeated 20 times
    risks_list, ci = n_times_k_cross_validation(20, X, y, 10, h_0, true_risk_proxy, random_seed=2000+i)
    saving_dict['20-10-fold'][0] += risks_list
    saving_dict['20-10-fold'][1].append(ci)

    print(f"Progress: {(i + 1) / 0.5}%", end='\r')


# 4. Compute the results

# Plot densities for each estimator
for (key, value) in saving_dict.items():
    plt.figure(key)
    sns.displot([(x - true_risk_proxy) for x in value[0]], kind="kde")
    plt.title(f"Estimator: {name_key[key]}")
    plt.xlabel("estimated risk - true risk")
    plt.ylabel("density")
    plt.savefig(f"homework-02/plots/estimator_{key}.png")
    plt.show()

# Compute the true risk proxy
print(f"True risk proxy: {true_risk_proxy}")

for (key, value) in saving_dict.items():
    print('----------------------------------------')
    print(f"Estimator: {name_key[key]}")

    # Compute the mean difference
    print(f"Mean difference: {np.mean(value[0]) - true_risk_proxy}")
    # Compute the median standard error of the risk estimates
    print(f"Median standard error: {median_standard_error(value[0])}")
    # Compute % of times the 95% confidence interval contains the true risk proxy
    print(f"Confidence interval contains true risk proxy: {np.mean(value[1]) * 100}%")
