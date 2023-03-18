import numpy as np
import pandas as pd
from aux_functions import toy_data, model_risks
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    huge_dataset = toy_data(100000, seed=0)
    huge_X = np.vstack(huge_dataset['x'].to_numpy())
    huge_y = np.array(huge_dataset['y'])

    diff = []

    for j in range(50):

        # 1. generate two toy datasets with 50 observations each
        toy_dataset1 = toy_data(50, seed=(1 + (2 * j)))
        toy_dataset2 = toy_data(50, seed=(2 + (2 * j)))

        # 2. Train model h1 using a learner and the first dataset only
        X1 = np.vstack(toy_dataset1['x'].to_numpy())
        y1 = np.array(toy_dataset1['y'])

        h_1 = LogisticRegression()
        h_1.fit(X1, y1)

        # 3. Train model h2 using the same learner and both datasets combined
        # for a total of 100 training observations.
        concat_X = pd.concat([toy_dataset1['x'],
                              toy_dataset2['x']])
        concat_y = pd.concat([toy_dataset1['y'],
                              toy_dataset2['y']])

        X2 = np.vstack(concat_X.to_numpy())
        y2 = np.array(concat_y)

        h_2 = LogisticRegression()
        h_2.fit(X2, y2)

        # 4. Compute true risk proxies for both models using the huge dataset
        probabilities_1 = h_1.predict_proba(huge_X)
        true_risks_1 = model_risks(huge_y, probabilities_1)
        true_risk_proxy_1 = np.mean(true_risks_1)

        probabilities_2 = h_2.predict_proba(huge_X)
        true_risks_2 = model_risks(huge_y, probabilities_2)
        true_risk_proxy_2 = np.mean(true_risks_2)

        diff.append(true_risk_proxy_1 - true_risk_proxy_2)

    # 6. Compute a summary of difference between the two models
    min_value = np.min(diff)
    first_quartile = np.percentile(diff, 25)
    median = np.median(diff)
    mean = np.mean(diff)
    third_quartile = np.percentile(diff, 75)
    max_value = np.max(diff)

    # Print results
    print(f"Minimum: {round(min_value, 4)}")
    print(f"1st Quantile: {round(first_quartile, 4)}")
    print(f"Median: {round(median, 4)}")
    print(f"Mean: {round(mean, 4)}")
    print(f"3rd Quantile: {round(third_quartile, 4)}")
    print(f"Maximum: {round(max_value, 4)}")
