# Ridge, Lasso

Implement ridge and lasso regression. Both model should fit the intercept without penalization. Use a closed-form solution for ridge regression and any optimization method for lasso (for example, the Powell method from `scipy`).

Implement models as classes (`RidgeReg`, `LassoReg`) which take a regularization weight as their constructor parameter and provide methods `fit(X, y)` and `predict(X)`. Objects `X` and `y` are numpy arrays, fit fits the model and doesn't return anything, while predict returns predicted values. Add your own tests for testing the correctness and expected behavior of your solution.

## Application

Apply your Ridge regression model to the provided superconductivity data set (`superconductor.csv`). It contains data about different superconducting materials with the goal of modeling their critical temperatures. The data set was somewhat adapted for the the purposes of this homework. Your goal is to minimize the model's *root mean square error* (RMSE). Use the first 200 examples as your training data for building the model and selecting an appropriate regularization weight. Note that you have relatively few data samples. Finally, estimate the model's *RMSE* on the other 100 examples.

Submit two files: a short (max 2 pages) report (`.pdf`) and your code (a single `Python 3.8`-compatible file; your code should only run things under `if __name__ == "__main__"`).