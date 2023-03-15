import numpy as np

from data_gen import toy_data, log_loss
from sklearn.linear_model import LogisticRegression

# we investigate how our test data risk estimate varies with test data
# 1. generate toy dataset
toy_dataset = toy_data(50, seed=1)
huge_dataset = toy_data(100000, seed=0)

# 2. train model h, using linear regression
X = np.vstack(toy_dataset['x'].to_numpy())
y = np.array(toy_dataset['y'])

h = LogisticRegression().fit(X, y)

# 3. 
# compute the true risk proxy using huge_dataset
huge_X = np.vstack(huge_dataset['x'].to_numpy())
huge_y = np.array(huge_dataset['y'])
huge_probabilities = h.predict_proba(huge_X)

def model_risks(y, probabilities, classes):
    """
    Calculates expected negative log loss for the model `h`
    """
    model_risks = []
    for i in range(len(y)):
        model_risks.append(log_loss(y[i], probabilities[i][classes == y[i]]))
    return model_risks

# compute the true risk proxy using huge_dataset
# true_risk_proxy = model_risk(huge_y, huge_probabilities, h.classes_)
true_risk_proxy = 0.93415131


# 4. For 1000 times generate toy dataset with 50 samples, estimate the risk of
#    the model h, and compute the standard error of the estimate and record
#    if the 95% confidence interval contains the true risk proxy

def standard_error_of_model_risk(risks):
    """
    Returns the standard error of the model risk estimate
    """
    return np.std(risks)

def confidence_interval_of_model_risk(risks):
    """
    Returns the 95% confidence interval of the model risk
    """
    return (np.mean(risks) - 1.96 * standard_error_of_model_risk(risks), 
            np.mean(risks) + 1.96 * standard_error_of_model_risk(risks))
    
def average_model_risk(risks):
    """
    Returns the average model risk
    """
    return np.mean(risks)



all_risks = []

for i in range(1000):
    toy_dataset = toy_data(50, seed=i)
    X = np.vstack(toy_dataset['x'].to_numpy())
    y = np.array(toy_dataset['y'])
    probabilities = h.predict_proba(X)
    risks = model_risks(y, probabilities, h.classes_)
    ci = confidence_interval_of_model_risk(risks)

    if true_risk_proxy > ci[0] and true_risk_proxy < ci[1]:
        all_risks.append(average_model_risk(risks))


# 5. Find results
differences = np.array(all_risks) - np.array(len(all_risks) * [true_risk_proxy])

# Plot the density of differences
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.displot(differences, kind='kde')
plt.title('Distribution of risk differences')
plt.xlabel('est_risk -  true_risk')
plt.ylabel('Density')
plt.savefig('homework-02/risk_differences.png')

# Compute the true risk proxy
true_risk_proxy = 0.93415131

# Compute the average difference between the estimate and true risk
average_difference = np.mean(differences)

# Compute the true risk of always making 0.5 - 0.5 predictions


# Compute the median standard error of the estimates


# Compute the % of times the true risk proxy is in the 95% confidence interval


def mse(y, y_hat):
    """
    Returns the mean squared error between y and y_hat
    """
    return np.mean((y - y_hat) ** 2)



