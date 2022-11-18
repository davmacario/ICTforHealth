import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def normalizeData(X, mean_X, stdev_X):
    X_norm = (X - mean_X)/stdev_X

    return X_norm


class GaussianProcessRegression():

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
        tmp_R = np.zeros((self.Np_tr+1, self.Np_tr+1))

        X_total = np.vstack([self.X_tr_norm, self.X_te_norm])

        # Possible to improve - the matrix to be built is actually symmetric
        for n in range(self.Np_tr+1):
            for k in range(self.Np_tr+1):
                tmp_R[n, k] = self.theta * \
                    np.exp(np.linalg.norm(
                        X_total[n, :] - X_total[k, :])/(2*self.r2))

        self.R_N = tmp_R + self.var_nu*np.identity(self.Np_tr+1)

        self.R_Nmin1 = self.R_N[:-1, :-1]
        self.k = self.R_N[:-1, -1]
        self.d = self.R_N[-1, -1].item()

    def solve(self):
        if ((self.R_N == 0).all()):
            self.eval_params()

        R_N_1_inv = np.linalg.inv(self.R_Nmin1)
        k = self.k[:, np.newaxis]
        y_tr_norm = self.y_tr_norm[:, np.newaxis]

        self.mu = (k.T@R_N_1_inv@y_tr_norm).item()
        self.var = (self.d - k.T@R_N_1_inv@k).item()

        self.y_hat_te = self.mu*self.std_y + self.mean_y

        return self.y_hat_te
