# https://github.com/craig-m-k/Recursive-least-squares/blob/master/RLS.ipynb

import numpy as np
import math

import pdb

class RLS:
    def __init__(self, num_vars, lam, delta):
        '''
        num_vars: number of variables including constant
        lam: forgetting factor, usually very close to 1.
        '''
        self.num_vars = num_vars

        # delta controls the initial state.
        self.A = delta * np.matrix(np.identity(self.num_vars))
        self.w = np.matrix(np.zeros(self.num_vars))
        self.w = self.w.reshape(self.w.shape[1], 1)

        # Variables needed for add_obs
        self.lam_inv = lam ** (-1)
        self.sqrt_lam_inv = math.sqrt(self.lam_inv)

        # A priori error
        self.a_priori_error = 0

        # Count of number of observations added
        self.num_obs = 0

    def add_obs(self, x, t):
        '''
        Add the observation x with label t.
        x is a column vector as a numpy matrix
        t is a real scalar
        '''
        x = np.matrix(x).T
        z = self.lam_inv * self.A * x
        alpha = float((1 + x.T * z) ** (-1))
        self.a_priori_error = float(t - self.w.T * x)
        self.w = self.w + (t - alpha * float(x.T * (self.w + t * z))) * z
        self.A -= alpha * z * z.T
        self.num_obs += 1

    def fit(self, X, y):
        '''
        Fit a model to X,y.
        X and y are numpy arrays.
        Individual observations in X should have a prepended 1 for constant coefficient.
        '''
        for i in range(len(X)):
            x = np.transpose(np.matrix(X[i]))
            self.add_obs(x, y[i])

    def get_error(self):
        '''
        Finds the a priori (instantaneous) error.
        Does not calculate the cumulative effect
        of round-off errors.
        '''
        return self.a_priori_error

    def predict(self, x):
        '''
        Predict the value of observation x. x should be a numpy matrix (row vector)
        '''
        x = np.matrix(x)
        predicted_ys = (self.w.T * x.T).T
        return np.squeeze(np.asarray(predicted_ys))