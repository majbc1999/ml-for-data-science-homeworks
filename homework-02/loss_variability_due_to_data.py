import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aux_functions import (toy_data, model_risks,
                           confidence_interval_of_model_risk,
                           median_standard_error, calculate_baseline_true_risk,
                           is_in_ci)
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
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

    # compute the true risk proxy using huge_dataset
    true_risks = model_risks(huge_y, huge_probabilities)
    true_risk_proxy = np.mean(true_risks)

    # 4. For 1000 times generate toy dataset with 50 samples, estimate the risk
    #    of the model h, and compute the standard error of the estimate and
    #    record if the 95% confidence interval contains the true risk proxy

    all_risks = []

    for i in range(1000):
        generated_toy_dataset = toy_data(50, seed=2+i)
        X = np.vstack(generated_toy_dataset['x'].to_numpy())
        y = np.array(generated_toy_dataset['y'])
        probabilities = h.predict_proba(X)
        risks = model_risks(y, probabilities)

        ci = confidence_interval_of_model_risk(risks)

        if true_risk_proxy > ci[0] and true_risk_proxy < ci[1]:
            all_risks.append((np.mean(risks), True))
        else:
            all_risks.append((np.mean(risks), False))

    # 5. Find results
    differences = np.array(all_risks)[:, 0] - \
        np.array(len(all_risks) * [true_risk_proxy])

    # Plot the density of differences
    plt.figure(figsize=(8, 6))
    sns.displot(differences, kind='kde')
    plt.title('Distribution of risk differences')
    plt.xlabel('est_risk -  true_risk')
    plt.ylabel('Density')
    plt.savefig('homework-02/plots/risk_differences_1.png',
                bbox_inches='tight')

    # Compute the true risk proxy
    print(f'True risk proxy: {true_risk_proxy}')

    # Compute the average difference between the estimate and true risk
    average_difference = np.mean(differences)
    print(f'Mean difference: {abs(average_difference)}')

    # Compute the true risk of always making 0.5 - 0.5 predictions
    true_risk_of_05 = calculate_baseline_true_risk(true_risks, true_risk_proxy)
    print(f'0.5-0.5 baseline true risk: {true_risk_of_05}')

    # Compute the median standard error of the estimates
    med_se = median_standard_error(np.array(risks))
    print(f'Median standard error: {med_se}')

    # Compute the % of times the true risk proxy is in the
    # 95% confidence interval
    percentage = is_in_ci(all_risks)
    print(f'Percentage of 95CI that contain the true risk proxy: {percentage}')
