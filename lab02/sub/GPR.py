import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def normalizeData(X, mean_X, stdev_X):
    X_norm = (X - mean_X)/stdev_X

    return X_norm


class GaussianProcessRegression():
    """
    Class used to solve Gaussian Process Regression problems.
    The solution is based on the assumption that the samples of 
    the process are correlated via a gaussian-like 
    autocorrelation.
    -------------------------------------------------------------
    Input parameters:
    - `y_train`: training regressand vector
    - `X_train`: training regressors matrix (each row is a 
    regressor)
    - `y_test`: test regressand (scalar)
    - `X_test`: test regressor (1-D vector)
    - `r_2`: value of the hyperparameter r^2 (exp. denominator in
    the gaussian autocorrelation function)
    - `theta`: value of the hyperparameter of the coefficient in
    the autocorrelation expression
    - `var_nu`: hyperparameter corresponding to the variance of 
    the measurement noise
    - `normalize` (default: True): bool to indicate whether to
    normalize the given data
    -------------------------------------------------------------
    """

    def __init__(self, y_train, X_train, y_test, X_test, r_2, theta, var_nu, normalize=True):
        self.Np_tr, self.Nf = X_train.shape

        if (len(X_test) != self.Nf):
            raise ValueError(
                "Error: the tested element has a wrong number of elements!\n")

        self.y_tr = y_train
        self.X_tr = X_train

        self.R_N = np.zeros((self.Np_tr+1, self.Np_tr+1))
        self.R_Nmin1 = np.zeros((self.Np_tr, self.Np_tr))
        self.k = np.zeros((self.Np_tr,))
        self.d = 0

        self.y_te = y_test
        self.X_te = X_test

        self.r2 = r_2
        self.theta = theta
        self.var_nu = var_nu

        self.y_hat_te = 0
        self.mu = 0
        self.var = 0

        if (normalize):
            self.mean_X = np.mean(X_train)
            self.std_X = np.std(X_train)
            self.X_tr_norm = normalizeData(X_train, self.mean_X, self.std_X)
            self.X_te_norm = normalizeData(X_test, self.mean_X, self.std_X)

            self.mean_y = np.mean(y_train)
            self.std_y = np.std(y_train)
            self.y_tr_norm = normalizeData(y_train, self.mean_y, self.std_y)
            self.y_te_norm = normalizeData(y_test, self.mean_y, self.std_y)

        else:
            self.mean_X = np.zeros((self.Nf, ))
            self.std_X = np.ones((self.Nf, ))

            self.mean_y = 0
            self.std_y = 1

            self.X_tr_norm = self.X_tr
            self.X_te_norm = self.X_te
            self.y_tr_norm = self.y_tr
            self.y_te_norm = self.y_te

    def eval_params(self):
        """
        Used to evaluate the parameters needed for performing GPR.
        -------------------------------------------------------------
        In particular, the evaluated quantities are: 
        - Full covariance matrix R_N, which includes the N-1 x N-1 
        matrix used for evaluating mean and variance of the predicted 
        sample(s)
        - Covariance matrix R_N-1
        - Vector k
        - Scalar value d (element in position N,N of the full 
        covariance matrix)
        -------------------------------------------------------------
        """
        tmp_R = np.zeros((self.Np_tr+1, self.Np_tr+1))

        X_total = np.vstack([self.X_tr_norm, self.X_te_norm])

        # Optimize execution since the matrix is symmetric
        for n in range(self.Np_tr+1):
            for k in range(n, self.Np_tr+1):
                # Evaluate the whole covariance matrix (including tested element)
                # The norm of the distance between every pair of regressing points (x)
                tmp_R[n, k] = self.theta * \
                    np.exp(np.linalg.norm(
                        X_total[n, :] - X_total[k, :])/(2*self.r2))
                tmp_R[k, n] = tmp_R[n, k]

        self.R_N = tmp_R + self.var_nu*np.identity(self.Np_tr+1)

        self.R_Nmin1 = self.R_N[:-1, :-1]
        self.k = self.R_N[:-1, -1]
        self.d = self.R_N[-1, -1].item()

    def solve(self):
        """
        Solve the Gaussian Process Regression problem, i.e., estimate
        the value of the regressand as the estimate mean value of the 
        probability density function
        -------------------------------------------------------------
        This method calls the evaluation of the parameters 
        (`self.eval_params()`) if it detects that they have not been 
        computed yet
        """
        if ((self.R_N == 0).all()):
            self.eval_params()

        R_N_1_inv = np.linalg.inv(self.R_Nmin1)
        k = self.k[:, np.newaxis]
        y_tr_norm = self.y_tr_norm[:, np.newaxis]

        self.mu = (k.T@R_N_1_inv@y_tr_norm).item()
        self.var = (self.d - k.T@R_N_1_inv@k).item()

        self.y_hat_te = self.mu*self.std_y + self.mean_y

        return self.y_hat_te
