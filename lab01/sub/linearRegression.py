import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sub.minimization as mymin


class LinearRegression():
    """ 
    This class is used to handle linear regression problems by using 
    LLS or Steepest Descent
    -----------------------------------------------------------------------------------
    Parameters:
    - regressand: Pandas DataFrame containing the regressand
    - regressors: Pandas Dataframe containing the regressors
    -----------------------------------------------------------------------------------
    Attributes:
    - regressand: regressand vector - Np elements (np.Ndarray)
    - regressors: regressors matrix - Np x Nf (np.Ndarray)
    - regressing_features: list of the features of the regressors
    - regressand_name: name of the regressand feature
    - w_hat_LLS: solution of the linear regression using Linear Least Squares method
    - w_hat_SD: solution of the linear regression using Steepest Descent algorithm
    - y_hat_LLS: approximated regressand using LLS solution
    - y_hat_SD: approximated regressand using SD solution
    - y_hat_LLS_norm: normalized LLS solution
    - y_hat_SD_norm: normalized SD solution
    - mean_regressors: mean of the regressing features
    - stdev_regressors: standard deviation of the regressing features
    - regressors_norm: normalized regressors matrix (zero mean, unit variance)
    - mean_regressand: mean of the (actual) regressand
    - stdev_regressand: standard deviation of the regressand
    - regressand_norm: normalized (training) regressand
    - LLS_error_train: absolute error of the LLS solution (over the training set)
    - SD_error_train: absolute error of the SD solution (over the training set)
    -----------------------------------------------------------------------------------
    """

    def __init__(self, regressand, regressors, test_regressand, test_regressors):
        # Check validity ()
        self.Np, self.Nf = regressors.values.shape
        if (regressand.values.shape[0] != self.Np):
            raise ValueError(
                "The dimensions of regressand and regressors are not coherent!\n")

        if (len(regressand.values.shape) > 1 and regressand.values.shape[1] > 1):
            raise ValueError("The regressand is not a 1D vector!\n")

        if (test_regressand.values.shape != regressand.values.shape):
            raise ValueError(
                "The dimensions of the test regressand are different from the ones of the training set\n")

        if (test_regressors.values.shape != regressors.values.shape):
            raise ValueError(
                "The dimensions of the test regressors are different from the ones of the training set\n")

        # Training set

        self.regressand = regressand.values
        self.regressors = regressors.values

        self.regressing_features = list(regressors.columns)
        self.regressand_name = str(regressand.columns)

        # Initialize solutions
        self.w_hat_LLS = np.zeros((self.Nf,))
        self.w_hat_SD = np.zeros((self.Nf,))

        # Initialize approximated regressands
        self.y_hat_LLS = np.zeros((self.Np,))
        self.y_hat_SD = np.zeros((self.Np,))

        # Initialize normalized approximated regressands
        self.y_hat_LLS_norm = np.zeros((self.Np,))
        self.y_hat_SD_norm = np.zeros((self.Np,))

        # Define normalized values (on which the algorithm(s) will be performed)
        self.mean_regressors = self.regressors.mean(axis=0)
        self.stdev_regressors = self.regressors.std(axis=0)

        if (self.mean_regressors != 0 or self.stdev_regressors != 1):
            # Normalize
            self.regressors_norm = (
                self.regressors - self.mean_regressors)/self.stdev_regressors
        else:
            self.regressors_norm = self.regressors

        self.mean_regressand = self.regressand.mean(axis=0)
        self.stdev_regressand = self.regressand.std(axis=0)

        if (self.mean_regressand != 0 or self.stdev_regressand != 1):
            # Normalize
            self.regressand_norm = (
                self.regressand - self.mean_regressand)/self.stdev_regressand
        else:
            self.regressand_norm = self.regressand

        self.LLS_error_train = np.zeros((self.Np,))
        self.SD_error_train = np.zeros((self.Np,))

# TO BE REMOVED
        ## Test set ###################################################
        self.test_regressand = test_regressand.values
        self.test_regressors = test_regressors.values

        # Initialize approximated regressands (test)
        self.y_hat_LLS_test = np.zeros((self.Np,))
        self.y_hat_SD_test = np.zeros((self.Np,))

        # Initialize normalized approximated regressands (test)
        self.y_hat_LLS_norm_test = np.zeros((self.Np,))
        self.y_hat_SD_norm_test = np.zeros((self.Np,))

        self.test_regressand_norm = (
            self.test_regressand - self.mean_regressand)/self.stdev_regressand
        self.test_regressors_norm = (
            self.test_regressors - self.mean_regressors)/self.stdev_regressors

        # IDEA: non includere il test set come attributi, ma fare una
        # funzione che lo riceva come argomento a calcoli i risultati
        #
        # (Vedi LLS_test e SD_test)
######

    def solve_LLS(self, plot_w=False, save_png=False, imagepath="./lab01/img/LLS-w_hat.png"):
        """
        Solution of the Linear Regression by means of the Linear Least Squares method.
        This function fills the attribute w_hat_LLS.
        -----------------------------------------------------------------------------------
        Optional parameters: 
        - plot_w: (default False) if True, a plot of the weights vector (w_hat_LLS) is 
          produced
        - save_png: (default False) if True, the image will be saved in the specified path
        - imagepath: (default: "./lab01/img/LLS-w_hat.png") path in which the image will 
          be stored
        -----------------------------------------------------------------------------------
        """
        X_tr_norm = self.regressors_norm
        y_tr_norm = self.regressand_norm
        self.w_hat_LLS = np.linalg.inv(
            X_tr_norm.T@X_tr_norm)@(X_tr_norm.T@y_tr_norm)

        self.y_hat_LLS_norm = X_tr_norm@(self.w_hat_LLS)
        self.y_hat_LLS = self.stdev_regressand * \
            (self.y_hat_LLS_norm) + self.mean_regressand

        self.LLS_error_train = self.regressand - self.y_hat_LLS

        if plot_w:
            nn = np.arange(self.Nf)
            plt.figure(figsize=(6, 4))
            plt.plot(nn, self.w_hat_LLS, '-o')
            ticks = nn
            # Each tick will have the label of the corresp. regressor
            plt.xticks(ticks, self.regressing_features, rotation=90)
            plt.ylabel(r'$\^w(n)$')     # Using LaTex notation
            plt.title('LLS - Optimized weights')
            plt.grid()
            plt.tight_layout()
            if save_png:
                plt.savefig(imagepath)
            plt.show()

    def solve_SteepestDescent(self, Nit=50, plot_w=False, save_png=False, imagepath="./lab01/img/SD-w_hat.png"):
        """
        Solution of the Linear Regression by means of the Steepest Descent method.
        This function fills the attribute w_hat_SD.
        -----------------------------------------------------------------------------------
        Optional parameters: 
        - plot_w: (default False) if True, a plot of the weights vector (w_hat_SD) is 
          produced
        - save_png: (default False) if True, the image will be saved in the specified path
        - imagepath: (default: "./lab01/img/SD-w_hat.png") path in which the image will 
          be stored
        -----------------------------------------------------------------------------------
        """
        X_tr_norm = self.regressors_norm
        y_tr_norm = self.regressand_norm
        SD_problem = mymin.SteepestDescent(
            self.regressand_norm, self.regressors_norm)
        self.w_hat_SD = SD_problem.run(Nit)

        self.y_hat_SD_norm = X_tr_norm@(self.w_hat_SD)
        self.y_hat_LLS = self.stdev_regressand * \
            (self.y_hat_SD_norm) + self.mean_regressand

        self.SD_error_train = self.regressand - self.y_hat_SD

        if plot_w:
            nn = np.arange(self.Nf)
            plt.figure(figsize=(6, 4))
            plt.plot(nn, self.w_hat_SD, '-o')
            ticks = nn
            # Each tick will have the label of the corresp. regressor
            plt.xticks(ticks, self.regressing_features, rotation=90)
            plt.ylabel(r'$\^w(n)$')     # Using LaTex notation
            plt.title('Steepest Descent - Optimized weights')
            plt.grid()
            plt.tight_layout()
            if save_png:
                plt.savefig(imagepath)
            plt.show()

    def plot_w(self, save_png=False, imagepath="./lab01/img/w_hat_comparison.png"):
        """
        This mathod produces a comparison plot between the weights vectors 
        w_hat obtained with LLS and SD.
        -----------------------------------------------------------------------------------
        Optional parameters:
        - save_png: (default False) if True, the image will be saved in the specified path
        - imagepath: (default: "./lab01/img/w_hat_comparison.png") path in which the image 
          will be stored
        -----------------------------------------------------------------------------------
        """
        null_vect = np.zeros((self.Nf,))
        if (self.w_hat_LLS == null_vect or self.w_hat_SD == null_vect):
            print("Error! The values of w_hat have not been all computed yet!\n")
            return

        nn = np.arange(self.Nf)
        plt.figure(figsize=(6, 4))
        plt.plot(nn, self.w_hat_LLS, '-o', label="LLS")
        plt.plot(nn, self.w_hat_SD, '-o', label="Steepest Descent")
        ticks = nn
        # Each tick will have the label of the corresp. regressor
        plt.xticks(ticks, self.regressing_features, rotation=90)
        plt.ylabel(r'$\^w(n)$')     # Using LaTex notation
        plt.title('Optimized weights - comparison')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        if save_png:
            plt.savefig(imagepath)
        plt.show()

    def LLS_test(self, test_regressand, test_regressors, plot_hist=False, save_png=False, img_path='./lab01/img/LLS-err_hist.png'):
        """
        This method is used to estimate a test regressand given the test 
        regressors and using the weights evaluated with the LLS method
        -----------------------------------------------------------------------------------
        Parameters:
        - test_regressand: (Np,) vector
        - test_regressors: (Np, Nf) matrix
        -----------------------------------------------------------------------------------
        Optional parameters
        - plot_hist: (default False) if True, a histogram of the error values 
          (test_regressand - y_hat_LLS) will be produced
        - save_png: (default False) if True, the plot will be saved in the specified path
        - img_path: (default './lab01/img/LLS-err_hist.png') path at which the histogram 
          will be saved
        -----------------------------------------------------------------------------------
        """
        # Check w_hat_LLS already computed
        if (self.w_hat_LLS == np.zeros((self.Nf,))):
            self.solveLLS()

        ## Test set ###################################################
        y_test = test_regressand.values
        X_test = test_regressors.values

        # Normalize
        X_test_norm = (X_test - self.mean_regressors)/self.stdev_regressors

        # Obtain approximated regressand
        y_hat_LLS_norm_test = X_test_norm@self.w_hat_LLS
        # De-normalize
        y_hat_LLS_test = y_hat_LLS_norm_test*self.stdev_regressand + self.mean_regressand

        # Error
        err_LLS_test = y_test - y_hat_LLS_test

        if plot_hist:
            e = [self.LLS_error_train, err_LLS_test]

            plt.figure(figsize=(6, 4))
            plt.hist(e, bins=50, density=True, histtype='bar',
                     label=['training', 'test'])
            plt.xlabel(r"$e = y - \^y$")
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title('LLS - Error histogram')
            plt.tight_layout()
            if save_png:
                plt.savefig(img_path)
            plt.show()

        return err_LLS_test

    def SD_test(self, test_regressand, test_regressors, plot_hist=False, save_png=False, img_path='./lab01/img/SD-err_hist.png'):
        """
        This method is used to estimate a test regressand given the test 
        regressors and using the weights evaluated with the SD method
        -----------------------------------------------------------------------------------
        Parameters:
        - test_regressand: (Np,) vector
        - test_regressors: (Np, Nf) matrix
        -----------------------------------------------------------------------------------
        Optional parameters
        - plot_hist: (default False) if True, a histogram of the error values 
          (test_regressand - y_hat_SD) will be produced
        - save_png: (default False) if True, the plot will be saved in the specified path
        - img_path: (default './lab01/img/SD-err_hist.png') path at which the histogram 
          will be saved
        -----------------------------------------------------------------------------------
        """
        # Check w_hat_SD already computed
        if (self.w_hat_SD == np.zeros((self.Nf,))):
            self.solveSD()

        ## Test set ###################################################
        y_test = test_regressand.values
        X_test = test_regressors.values

        # Normalize
        X_test_norm = (X_test - self.mean_regressors)/self.stdev_regressors

        # Obtain approximated regressand
        y_hat_LLS_norm_test = X_test_norm@self.w_hat_SD
        # De-normalize
        y_hat_LLS_test = y_hat_LLS_norm_test*self.stdev_regressand + self.mean_regressand

        # Error
        err_SD_test = y_test - y_hat_LLS_test

        if plot_hist:
            e = [self.SD_error_train, err_SD_test]

            plt.figure(figsize=(6, 4))
            plt.hist(e, bins=50, density=True, histtype='bar',
                     label=['training', 'test'])
            plt.xlabel(r"$e = y - \^y$")
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title('SD - Error histogram')
            plt.tight_layout()
            if save_png:
                plt.savefig(img_path)
            plt.show()

        return err_SD_test

    def test(self, test_regressand, test_regressors, plot_hist=False, save_png=False, img_path='./lab01/img/err_hist_compare.png'):
        """
        This method is used to compare the performance of Linear Regression carried out
        with either LLS or Steepest Descent in terms of error on the regressand
        -----------------------------------------------------------------------------------
        Parameters:
        - test_regressand: (Np,) vector
        - test_regressors: (Np, Nf) matrix
        -----------------------------------------------------------------------------------
        Optional parameters
        - plot_hist: (default False) if True, a histogram of the error values 
          (test_regressand - y_hat_LLS) will be produced
        - save_png: (default False) if True, the plot will be saved in the specified path
        - img_path: (default './lab01/img/err_hist_compare.png') path at which the 
          histogram will be saved
        -----------------------------------------------------------------------------------
        Returned variable(s):
        - e: list containing the error vectors of LLS and SD over the test set
        -----------------------------------------------------------------------------------
        """
        # Call both LLS_test and SD_test and store the results (absolute errors) in order
        # to make the comparison between the two methods

        err_LLS_test = self.LLS_test(test_regressand, test_regressors)
        err_SD_test = self.SD_test(test_regressand, test_regressors)

        e = [err_LLS_test, err_SD_test]

        if plot_hist:
            plt.figure(figsize=(6, 4))
            plt.hist(e, bins=50, density=True, histtype='bar',
                     label=['LLS', 'SD'])
            plt.xlabel(r"$e = y - \^y$")
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title('Error histogram - comparison')
            plt.tight_layout()
            plt.savefig(img_path)
            plt.show()

        return e


if __name__ == '__main__':
    pass
