import numpy as np
import pandas as pd
from typing import Tuple, List
import pyjags

# classes

class BayesianLogisticRegression:
    """
    Bayesian logistic regression model.
    """

    # Thank the lord for this article
    # https://towardsdatascience.com/introduction-to-bayesian-logistic-regression-7e39a0bae691#:~:text=Bayesian%20logistic%20regression%20has%20the,of%20contraceptives%20usage%20per%20district.


# misc

def import_dataset(path: str) -> Tuple[np.ndarray, np.array]:
    """
    Function for importation of the dataset
    """
    df = pd.read_csv(path)
    
    y = df['Made'].to_numpy()
    X = df.drop('Made', axis=1).to_numpy()
    
    return X, y


if __name__ == '__main__':
    X, y = import_dataset('homework-06/dataset.csv')
    model = BayesianLogisticRegression()
    model.fit(X, y)
    predictions = model.predict(X[:5])
    print(predictions)
    print(y[:5])


