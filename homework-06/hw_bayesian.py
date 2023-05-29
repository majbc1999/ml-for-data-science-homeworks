import numpy as np
import pandas as pd
from typing import Tuple
import pymc3 as pm
from scipy.special import expit
import matplotlib.pyplot as plt

# classes

class BayesianLogisticRegression:
    """
    Bayesian logistic regression model.
    """

    def __init__(self, n_samples: int = 10000, normalize: bool = False):
        """
        Initialize the model with `n_samples` posterior samples.
        """
        self.n_samples = n_samples
        self.normalize = normalize

        self._data_mean = None
        self._data_std = None

        self.trace = None
        self.model = None


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.
        """
        if self.normalize:
            self._data_mean = np.mean(X, axis=0)
            self._data_std = np.std(X, axis=0)
            X = (X - self._data_mean) / self._data_std

        with pm.Model() as logistic_model:
            # Priors
            intercept = pm.Normal("intercept", mu=0, sd=10)
            beta1 = pm.Normal("beta1", mu=0, sd=10)
            beta2 = pm.Normal("beta2", mu=0, sd=10)
            
             # Logistic regression equation
            logit_p = intercept + pm.math.dot([beta1, beta2], X.T)
            
            # Likelihood
            likelihood = pm.Bernoulli("likelihood", logit_p=logit_p, observed=y)
            
            # Sampling from the posterior distribution
            trace = pm.sample(self.n_samples, 
                              tune=1000, 
                              cores=1, 
                              return_inferencedata=False)

            self.trace = trace
            self.model = logistic_model

        return self


    def get_betas(self):
        """
        Return intercept and betas for sampling.
        """
        # Retrieve the posterior samples for the coefficients
        intercept_samples = self.trace["intercept"]
        beta1_samples = self.trace["beta1"]
        beta2_samples = self.trace["beta2"]

        return intercept_samples, beta1_samples, beta2_samples


    def predict_instance_distribution(self, 
                                      x: np.ndarray, 
                                      show_hist: bool = False) -> np.ndarray:
        """
        Predict distribution of probabilities for a single instance.
        """
        if self.trace is None:
            raise RuntimeError("The model has not been fitted. "
                               "Call the fit method first.")
        
        # Normalize the data
        if self.normalize:
            x = (x - self._data_mean) / self._data_std

        intercept, beta_1, beta_2 = self.get_betas()

        # Compute the logit for each sample from the posterior distribution
        probs = expit(intercept + beta_1 * x[0] + beta_2 * x[1])

        if show_hist:
            plt.figure()
            plt.hist(probs, bins=50, density=True)
            plt.title("Probability distribution")
            plt.show()

        return probs
        

    def print_plots(self, save: bool = False):
        """
        Print plots.
        """
        if self.trace is None:
            raise RuntimeError("The model has not been fitted. " 
                               "Call the fit method first.")
        
        intercept, beta_1, beta_2 = self.get_betas()

        plt.figure('Intercept distribution')
        plt.hist(intercept, bins=50, density=True)
        plt.title("Intercept distribution")
        if save:
            plt.savefig("homework-06/plots/intercept.png", bbox_inches='tight')

        plt.figure('Beta 1 distribution')
        plt.hist(beta_1, bins=50, density=True)
        plt.title("Beta 1 distribution (angle)") 
        if save:
            plt.savefig("homework-06/plots/beta_1.png", bbox_inches='tight')

        plt.figure('Beta 2 distribution')
        plt.hist(beta_2, bins=50, density=True)
        plt.title("Beta 2 distribution (distance)")
        if save:
            plt.savefig("homework-06/plots/beta_2.png", bbox_inches='tight')

        plt.show()
        return


    def statistics(self, save: bool = False):
        """
        Returns pandas dataframe with statistics
        """
        df = pd.DataFrame(columns=['coef', 'mean', 'std', '2.5%', '97.5%'])
        intercept, beta_1, beta_2 = self.get_betas()

        df.loc[0] = ['intercept', 
                     intercept.mean(), 
                     intercept.std(), 
                     np.quantile(intercept, 0.025), 
                     np.quantile(intercept, 0.975)]
        df.loc[1] = ['beta 1 (angle)', 
                     beta_1.mean(), 
                     beta_1.std(), 
                     np.quantile(beta_1, 0.025), 
                     np.quantile(beta_1, 0.975)]
        df.loc[2] = ['beta_2 (distance)', 
                     beta_2.mean(), 
                     beta_2.std(), 
                     np.quantile(beta_2, 0.025), 
                     np.quantile(beta_2, 0.975)]

        if save:
            df.to_csv('homework-06/plots/statistics.csv', index=False)

        return df

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

    # build a model and fit it (10000 posterior samples)
    model = BayesianLogisticRegression(n_samples=10000,
                                       normalize=True).fit(X, y)

    # print distribution plots
    model.print_plots(save=True)

    print('\n')

    # print statistics
    print('Statistics:')
    print(model.statistics(save=True))

