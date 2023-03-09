# Firstly, we have to convert this bit of code from R into Python
#
#   toy_data <- function(n, seed = NULL) {
#       set.seed(seed)
#       x <- matrix(rnorm(8 * n), ncol = 8)
#       z <- 0.4 * x[,1] - 0.5 * x[,2] + 1.75 * x[,3] - 0.2 * x[,4] + x[,5]
#       y <- runif(n) > 1 / (1 + exp(-z))
#       return (data.frame(x = x, y = y))
#   }
#   log_loss <- function(y, p) {
#       -(y * log(p) + (1 - y) * log(1 - p))
#   }

import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    # Generate the data
    df_dgp = toy_data(100000, 0)

    
