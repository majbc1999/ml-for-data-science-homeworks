import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aux_functions import toy_data, split_dataset, model_risks, \
    confidence_interval_of_model_risk, median_standard_error
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    # 1. Generate a toy dataset with 100 observations
    toy_dataset = toy_data(100, seed=1)

    # 2. Train model h0 using some learner on all 100 observations and compute
    # its true risk proxy using the huge dataset.

    X = np.vstack(toy_dataset['x'].to_numpy())
    y = np.array(toy_dataset['y'])
    h0 = LogisticRegression().fit(X, y)

    huge_dataset = toy_data(100000, seed=0)
    huge_X = np.vstack(huge_dataset['x'].to_numpy())
    huge_y = np.array(huge_dataset['y'])

    probabilities = h0.predict_proba(huge_X)

    true_risks = model_risks(huge_y, probabilities)
    true_risk_proxy = np.mean(true_risks)

    # 3. Split the dataset into 50 training and 50 test observations at random.

    # 4. Repeat 1000 times:
    count = 0
    risk_estimates = []

    for i in range(1000):
        toy_dataset1, toy_dataset2 = split_dataset(toy_dataset, seed=i)

        X_1 = np.vstack(toy_dataset1['x'].to_numpy())
        X_2 = np.vstack(toy_dataset2['x'].to_numpy())

        y_1 = np.array(toy_dataset1['y'])
        y_2 = np.array(toy_dataset2['y'])

        h = LogisticRegression().fit(X_1, y_1)

        probabilities_2 = h.predict_proba(X_2)
        risks = model_risks(y_2, probabilities_2)
        lb, hb = confidence_interval_of_model_risk(risks)
        if true_risk_proxy > lb and true_risk_proxy < hb:
            count += 1
        risk_estimates.append(np.mean(risks))

    # 6. Find results

    # Plot density estimate of the differences

    differences = np.array(risk_estimates) - true_risk_proxy

    plt.figure(figsize=(8, 6))
    sns.displot(differences, kind='kde')
    plt.title('Distribution of risk differences')
    plt.xlabel('est_risk - true_risk')
    plt.ylabel('Density')
    plt.savefig('homework-02/plots/risk_differences_2.png',
                bbox_inches='tight')

    ci_contains_true_risk = count / 1000

    print(f'True risk proxy: {true_risk_proxy}')
    print(f'Mean difference: {abs(np.mean(differences))}')
    med_se = median_standard_error(np.array(risks))
    print(f'Median standard error: {med_se}')
    print(f'Percentage of 95CI that contain the true risk proxy:'
          f' {100 * ci_contains_true_risk}')
