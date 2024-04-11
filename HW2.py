#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Exercise 0 
def github() -> str:
    """
    This is my HW1 (the link to github)
    """

    return "https://github.com/YantingHuMn/Econ481HW/blob/main/HW2.py"


# In[41]:


# Exercise 1
import numpy as np

def simulate_data(seed: int = 481) -> tuple:
    """
    This function is to return 1000 simulated observations for a linear regression model
    
    Parameters: 
    seed (int): the seed set to this function

    Result:
    Return a tuple of two elements, (y, X), where y is a 1000x1 np.array and X is a 1000x3 np.array.
    """
    np.random.seed(seed) 
    x1 = np.random.normal(loc = 0, scale=np.sqrt(2), size=(1000, 1))
    x2 = np.random.normal(loc = 0, scale=np.sqrt(2), size=(1000, 1))
    x3 = np.random.normal(loc = 0, scale=np.sqrt(2), size=(1000, 1))
    X = np.column_stack((x1, x2, x3)) # all i.i.d. variables
    epsilon = np.random.normal(loc = 0, scale = 1, size = 1000)
    y = 5 + 3*X[:,0] + 2*X[:,1] + 6*X[:,2] + epsilon
    return y.reshape(-1, 1), X

y, X = simulate_data()
#print(y)
#print(X)
print(simulate_data(481))


# In[58]:


# Exercise 2

import numpy as np
import scipy as sp

def estimate_mle(y: np.array, X: np.array) -> np.array:
    """
    This function is to calcualte the maximum likelihood estimator for all the 
    coefficients in the regression model

    Parameters:
    y (array): Response variables
    x (array): Predictors

    Result: 
    Return a 4 by 1 array with coefficients for beta in sequence
    """
    def log_likelihood(params):
        """
        This function is to calculate the log likelihood function

        Parameters:
        params: an array of 4 by 1 array with coefficients for beta in sequence

        Result:
        Return negative likelihood value
        """
        beta0, beta1, beta2, beta3 = params
        epsilon = y[:,0] - beta0 - beta1*X[:,0] - beta2*X[:,1] - beta3*X[:,2]
        log_likelihood = -0.5 * np.sum(np.log(2*np.pi) + np.log(1) + 1*(epsilon)**2)
        return -log_likelihood
    
    result = sp.optimize.minimize(log_likelihood, [1,1,1,1], method='Nelder-Mead')
    return np.array(result.x.reshape(-1,1))

y, X = simulate_data(481)
print(estimate_mle(y, X))


# In[59]:


# Ecercise 3
import numpy as np

def estimate_ols(y: np.array, X: np.array) -> np.array:
    """
    This function is to estimate OLS coefficients.

    Parameters:
    y (array): Response variables
    x (array): Predictors.

    Returns:
    Return a 4 by 1 array with coefficients for beta in sequence
    """
    def ols(params):
        """
        This function is to calculate the OLS function

        Parameters:
        params: an array of 4 by 1 array with coefficients for beta in sequence

        Result:
        Return the sum of squared residuals (OLS). 
        """
        beta0, beta1, beta2, beta3 = params
        epsilon = y[:,0] - beta0 - beta1*X[:,0] - beta2*X[:,1] - beta3*X[:,2]
        ols = np.sum(epsilon**2)
        return ols
        
    beta_hat = sp.optimize.minimize(ols, [1,1,1,1], method='Nelder-Mead')
    return np.array(beta_hat.x.reshape(-1,1))

y, X = simulate_data(481)
print(estimate_ols(y, X))


# In[ ]:




