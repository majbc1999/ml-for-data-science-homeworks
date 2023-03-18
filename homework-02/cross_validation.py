from aux_functions import (toy_data, model_risks, k_fold_cross_validation,
                           leave_one_out_cross_validation,
                           n_times_k_cross_validation, median_standard_error)
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    saving_dict = {
        '2-fold': [[], [], []],
        'leave-one-out': [[], [], []],
        '10-fold': [[], [], []],
        '4-fold': [[], [], []],
        '20-10-fold': [[], [], []]
    }

    name_key = {
        '2-fold': '2-fold cross validation',
        'leave-one-out': 'leave-one-out cross validation',
        '10-fold': '10-fold cross validation',
        '4-fold': '4-fold cross validation',
        '20-10-fold': '10-fold cross validation repeated 20 times'
    }

    for i in range(500):

        # 1. generate a toy dataset with 100 samples
        toy_dataset = toy_data(100, seed=1000+i)

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
        risk_difference, in_ci = k_fold_cross_validation(X, y, 2, h_0,
                                                         true_risk_proxy,
                                                         random_seed=2000+i)
        saving_dict['2-fold'][0].append(risk_difference)
        saving_dict['2-fold'][1].append(in_ci)
        saving_dict['2-fold'][2].append(risk_difference + true_risk_proxy)
        # b) leave-one-out cross validation
        risk_difference, in_ci = leave_one_out_cross_validation(X, y, h_0,
                                                                true_risk_proxy,
                                                                random_seed=2000+i)
        saving_dict['leave-one-out'][0].append(risk_difference)
        saving_dict['leave-one-out'][1].append(in_ci)
        saving_dict['leave-one-out'][2].append(
            risk_difference + true_risk_proxy)
        # c) 10-fold cross validation
        risk_difference, in_ci = k_fold_cross_validation(X, y, 10, h_0,
                                                         true_risk_proxy,
                                                         random_seed=2000+i)
        saving_dict['10-fold'][0].append(risk_difference)
        saving_dict['10-fold'][1].append(in_ci)
        saving_dict['10-fold'][2].append(risk_difference + true_risk_proxy)
        # d) 4-fold cross validation
        risk_difference, in_ci = k_fold_cross_validation(X, y, 4, h_0,
                                                         true_risk_proxy,
                                                         random_seed=2000+i)
        saving_dict['4-fold'][0].append(risk_difference)
        saving_dict['4-fold'][1].append(in_ci)
        saving_dict['4-fold'][2].append(risk_difference + true_risk_proxy)
        # e) 10-fold cross validation repeated 20 times
        risk_difference, in_ci = n_times_k_cross_validation(20, X, y, 10, h_0,
                                                            true_risk_proxy,
                                                            random_seed=2000+i)
        saving_dict['20-10-fold'][0].append(risk_difference)
        saving_dict['20-10-fold'][1].append(in_ci)
        saving_dict['20-10-fold'][2].append(risk_difference + true_risk_proxy)

        print(f"Progress: {(i + 1) / 5}%", end='\r')

    # 4. Compute the results

    # Plot densities for each estimator
    for (key, value) in saving_dict.items():
        plt.figure()
        sns.displot(value[0], kind="kde")
        plt.title(f"Estimator: {name_key[key]}")
        plt.xlim(-0.4, 0.4)
        plt.ylim(0, 6.5)
        plt.xlabel("estimated risk - true risk")
        plt.ylabel("density")
        plt.savefig(
            f"homework-02/plots/estimator_{key}.png", bbox_inches='tight')

    for (key, value) in saving_dict.items():
        print('----------------------------------------')
        print(f"Estimator: {name_key[key]}")

        # Compute the mean difference
        print(f"Mean difference: {np.mean(value[0])}")
        # Compute the median standard error of the risk estimates
        print(f"Median standard error: {median_standard_error(value[2])}")
        # Compute % of times the confidence interval contains the true risk proxy
        print(f"Confidence interval contains true risk "
              f"proxy: {np.mean(value[1]) * 100}%")
