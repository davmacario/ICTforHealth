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
    - normalize: (default True) specify whether we need to normalize the input 
      (training) dataset before computing linear regression
    -----------------------------------------------------------------------------------
    Attributes:
    - regressand: regressand vector - Np elements (np.Ndarray)
    - regressors: regressors matrix - Np x Nf (np.Ndarray)
    - regressing_features: list of the features of the regressors
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

    def __init__(self, regressand, regressors, normalize=True):
        # Check validity
        self.Np, self.Nf = regressors.values.shape
        if (regressand.values.shape[0] != self.Np):
            raise ValueError(
                "The dimensions of regressand and regressors are not coherent!\n")

        if (len(regressand.values.shape) > 1 and regressand.values.shape[1] > 1):
            raise ValueError("The regressand is not a 1D vector!\n")

        # Training set

        self.regressand = regressand.values
        self.regressors = regressors.values

        self.regressing_features = list(regressors.columns)
        # NOTE: y is a 'Series' object, not a DataFrame, since it only contains 1 column
        self.regressand_name = str(regressand.name)

        # Initialize solutions
        self.w_hat_LLS = np.zeros((self.Nf,))
        self.w_hat_SD = np.zeros((self.Nf,))

        # Initialize approximated regressands
        self.y_hat_LLS = np.zeros((self.Np,))
        self.y_hat_SD = np.zeros((self.Np,))

        # Initialize normalized approximated regressands
        self.y_hat_LLS_norm = np.zeros((self.Np,))
        self.y_hat_SD_norm = np.zeros((self.Np,))

        if normalize:
            # Define normalized values (on which the algorithm(s) will be performed)
            self.mean_regressors = self.regressors.mean(axis=0)
            self.stdev_regressors = self.regressors.std(axis=0)
            # WHAT TO DO WHEN STDEV=0? - set the standard deviation to a low value
            self.stdev_regressors[self.stdev_regressors == 0] = 1e-8

            if (0 in self.stdev_regressors):
                print("One of the standard deviations is 0")
                print(self.regressing_features)
                print(self.stdev_regressors)

            if (not all(self.mean_regressors == np.zeros((self.Nf,))) or not all(self.stdev_regressors == np.ones((self.Nf,)))):
                # Normalize
                self.regressors_norm = (
                    self.regressors - self.mean_regressors)/self.stdev_regressors
            else:
                # Do not normalize
                self.regressors_norm = self.regressors

            self.mean_regressand = self.regressand.mean(axis=0)
            self.stdev_regressand = self.regressand.std(axis=0)
            if self.stdev_regressand == 0:  # Case stdev == 0
                self.stdev_regressand = 1e-8

            if (self.mean_regressand != 0 or self.stdev_regressand != 1) and (normalize):
                # Normalize
                self.regressand_norm = (
                    self.regressand - self.mean_regressand)/self.stdev_regressand
            else:
                self.regressand_norm = self.regressand

        else:
            # No normalization - set all means to 0 and stdev to 1
            self.mean_regressors = np.zeros((1, len(self.regressing_features)))
            self.stdev_regressors = np.ones((1, len(self.regressing_features)))
            self.regressors_norm = self.regressors

            self.mean_regressand = 0
            self.stdev_regressand = 1
            self.regressand_norm = self.regressand

            # NOTE: regressors_norm and regressand_norm will not necessarily be normalized in this case

        # Init. error vectors
        self.LLS_error_train = np.zeros((self.Np,))
        self.SD_error_train = np.zeros((self.Np,))

    def solve_LLS(self, plot_w=False, save_w=False, imagepath_w="./img/LLS-w_hat.png", plot_y=False, save_y=False, imagepath_y="./img/LLS-y_vs_y_hat.png"):
        """
        Solution of the Linear Regression by means of the Linear Least Squares method.
        This function fills the attribute w_hat_LLS.
        -----------------------------------------------------------------------------------
        Optional parameters: 
        - plot_w: (default False) if True, a plot of the weights vector (w_hat_LLS) is 
          produced
        - save_w: (default False) if True, the image will be saved in the specified path
        - imagepath_w: (default: "./img/LLS-w_hat.png") path in which the image will 
          be stored
        - plot_y: (default False) if True, plot the comparison between the actual 
          regressand and the approximated one (y_hat) obtained using LLS (de-normalized)
        - save_y: (default False) if True, save the image in the specified path
        - imagepath_y: (default "./img/LLS-y_vs_y_hat.png") path in which to store
          the comparison between y and y_hat
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
            if save_w:
                plt.savefig(imagepath_w)
            plt.show()

        if plot_y:
            plt.figure(figsize=(6, 4))
            plt.plot(self.regressand, self.y_hat_LLS, '.')   # Place dots
            v = plt.axis()
            # Plot 45deg diagonal line
            plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
            plt.xlabel(r'$y$')
            plt.ylabel(r'$\^y$')
            plt.grid()
            plt.title("LLS - Training set")
            plt.tight_layout()
            if save_y:
                plt.savefig(imagepath_y)
            plt.show()

    def solve_SteepestDescent(self, stoppingCondition='iterations', Nit=50, plot_w=False, save_w=False, imagepath_w="./img/SD-w_hat.png", plot_y=False, save_y=False, imagepath_y="./img/SD-y_vs_y_hat.png"):
        """
        Solution of the Linear Regression by means of the Steepest Descent method.
        This function fills the attribute w_hat_SD.
        -----------------------------------------------------------------------------------
        Optional parameters: 
        - Nit: number of iterations (stopping condition) for the Steepest Descent algorithm
        - plot_w: (default False) if True, a plot of the weights vector (w_hat_SD) is 
          produced
        - save_w: (default False) if True, the image will be saved in the specified path
        - imagepath_w: (default: "./img/SD-w_hat.png") path in which the image will 
          be stored
        - plot_y: (default False) if True, plot the comparison between the actual 
          regressand and the approximated one (y_hat) obtained using SD (de-normalized)
        - save_y: (default False) if True, save the image in the specified path
        - imagepath_y: (default "./img/SD-y_vs_y_hat.png") path in which to store
          the comparison between y and y_hat
        -----------------------------------------------------------------------------------
        """
        X_tr_norm = self.regressors_norm
        y_tr_norm = self.regressand_norm
        SD_problem = mymin.SteepestDescent(
            self.regressand_norm, self.regressors_norm)
        self.w_hat_SD = SD_problem.run(
            stoppingCondition=stoppingCondition, Nit=Nit)

        self.y_hat_SD_norm = X_tr_norm@(self.w_hat_SD)
        self.y_hat_SD = self.stdev_regressand * \
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
            if save_w:
                plt.savefig(imagepath_w)
            plt.show()

        if plot_y:
            plt.figure(figsize=(6, 4))
            plt.plot(self.regressand, self.y_hat_SD, '.')   # Place dots
            v = plt.axis()
            # Plot 45deg diagonal line
            plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
            plt.xlabel(r'$y$')
            plt.ylabel(r'$\^y$')
            plt.grid()
            plt.title("SD - Training set")
            plt.tight_layout()
            if save_y:
                plt.savefig(imagepath_y)
            plt.show()

    def plot_w(self, save_png=False, imagepath="./img/w_hat_comparison.png"):
        """
        This mathod produces a comparison plot between the weights vectors 
        w_hat obtained with LLS and SD.
        -----------------------------------------------------------------------------------
        Optional parameters:
        - save_png: (default False) if True, the image will be saved in the specified path
        - imagepath: (default: "./img/w_hat_comparison.png") path in which the image 
          will be stored
        -----------------------------------------------------------------------------------
        """
        null_vect = np.zeros((self.Nf,))
        if (all(self.w_hat_LLS == null_vect) or all(self.w_hat_SD == null_vect)):
            print("Error! The values of w_hat have not been all computed yet!\n")
            self.solve_LLS()
            self.solve_SteepestDescent()

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

    def LLS_test(self, test_regressand, test_regressors, plot_hist=False, save_hist=False, imagepath_hist='./img/LLS-err_hist.png', plot_y=False, save_y=False, imagepath_y='./img/LLS_y_test_vs_y_hat_test.png'):
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
        - save_hist: (default False) if True, the plot will be saved in the specified path
        - imagepath_hist: (default './img/LLS-err_hist.png') path at which the 
          histogram will be saved
        - plot_y: (default False) if True, plot the comparison between the actual 
          regressand and the approximated one (y_hat) obtained using LLS (de-normalized)
        - save_y: (default False) if True, save the image in the specified path
        - imagepath_y: (default './img/LLS_y_test_vs_y_hat_test.png') path in which 
          to store the comparison between y and y_hat
        -----------------------------------------------------------------------------------
        Returned variable(s):
        - err_LLS_test: Ndarray of absolute error (regressand - y_hat_LLS)
        - y_hat_LLS_test: approximated regressand
        -----------------------------------------------------------------------------------
        """
        # Check w_hat_LLS already computed
        if (all(self.w_hat_LLS == np.zeros((self.Nf,)))):
            self.solveLLS()

        if (test_regressand.values.shape != self.regressand.shape):
            raise ValueError(
                "The dimensions of the test regressand are different from the ones of the training set\n")

        if (test_regressors.values.shape != self.regressors.shape):
            raise ValueError(
                "The dimensions of the test regressors are different from the ones of the training set\n")

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
            if save_hist:
                plt.savefig(imagepath_hist)
            plt.show()

        if plot_y:
            plt.figure(figsize=(6, 4))
            plt.plot(test_regressand, y_hat_LLS_test, '.')   # Place dots
            v = plt.axis()
            # Plot 45deg diagonal line
            plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
            plt.xlabel(r'$y$')
            plt.ylabel(r'$\^y$')
            plt.grid()
            plt.title("LLS - Test set")
            plt.tight_layout()
            if save_y:
                plt.savefig(imagepath_y)
            plt.show()

        return err_LLS_test, y_hat_LLS_test

    def SD_test(self, test_regressand, test_regressors, plot_hist=False, save_hist=False, imagepath_hist='./img/SD-err_hist.png', plot_y=False, save_y=False, imagepath_y='./img/SD_y_test_vs_y_hat_test.png'):
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
        - save_hist: (default False) if True, the plot will be saved in the specified path
        - imagepath_hist: (default './img/SD-err_hist.png') path at which the 
          histogram will be saved
        - plot_y: (default False) if True, plot the comparison between the actual 
          regressand and the approximated one (y_hat) obtained using SD (de-normalized)
        - save_y: (default False) if True, save the image in the specified path
        - imagepath_y: (default './img/SD_y_test_vs_y_hat_test.png') path in which 
          to store the comparison between y and y_hat
        -----------------------------------------------------------------------------------
        Returned variable(s):
        - err_SD_test: Ndarray of absolute error (regressand - y_hat_SD)
        - y_hat_SD_test: approssimated regressand
        -----------------------------------------------------------------------------------
        """
        # Check w_hat_SD already computed
        if (all(self.w_hat_SD == np.zeros((self.Nf,)))):
            self.solveSD()

        if (test_regressand.values.shape != self.regressand.shape):
            raise ValueError(
                "The dimensions of the test regressand are different from the ones of the training set\n")

        if (test_regressors.values.shape != self.regressors.shape):
            raise ValueError(
                "The dimensions of the test regressors are different from the ones of the training set\n")

        ## Test set ###################################################
        y_test = test_regressand.values
        X_test = test_regressors.values

        # Normalize
        X_test_norm = (X_test - self.mean_regressors)/self.stdev_regressors

        # Obtain approximated regressand
        y_hat_SD_norm_test = X_test_norm@self.w_hat_SD
        # De-normalize
        y_hat_SD_test = y_hat_SD_norm_test*self.stdev_regressand + self.mean_regressand

        # Error
        err_SD_test = y_test - y_hat_SD_test

        if plot_hist:
            e = [self.SD_error_train, err_SD_test]

            plt.figure(figsize=(6, 4))
            plt.hist(e, bins=50, density=True, histtype='bar',
                     label=['training set', 'test set'])
            plt.xlabel(r"$e = y - \^y$")
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title('SD - Error histogram')
            plt.tight_layout()
            if save_hist:
                plt.savefig(imagepath_hist)
            plt.show()

        if plot_y:
            plt.figure(figsize=(6, 4))
            plt.plot(test_regressand, y_hat_SD_test, '.')   # Place dots
            v = plt.axis()
            # Plot 45deg diagonal line
            plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
            plt.xlabel(r'$y$')
            plt.ylabel(r'$\^y$')
            plt.grid()
            plt.title("SD - Test set")
            plt.tight_layout()
            if save_y:
                plt.savefig(imagepath_y)
            plt.show()

        return err_SD_test, y_hat_SD_test

    def test(self, test_regressand, test_regressors, plot_hist=False, save_hist=False, imagepath_hist='./img/err_hist_compare.png'):
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
        - save_hist: (default False) if True, the plot will be saved in the specified path
        - imagepath_hist: (default './img/err_hist_compare.png') path at which the 
          histogram will be saved
        -----------------------------------------------------------------------------------
        Returned variable(s):
        - e: list containing the error vectors of LLS and SD over the test set
        -----------------------------------------------------------------------------------
        """
        # Call both LLS_test and SD_test and store the results (absolute errors) in order
        # to make the comparison between the two methods

        err_LLS_test, y_hat_LLS_te = self.LLS_test(
            test_regressand, test_regressors)
        err_SD_test, y_hat_SD_te = self.SD_test(
            test_regressand, test_regressors)

        e = [err_LLS_test, err_SD_test]
        y_hat_list = [y_hat_LLS_te, y_hat_SD_te]

        if plot_hist:
            plt.figure(figsize=(6, 4))
            plt.hist(e, bins=50, density=True, histtype='bar',
                     label=['LLS', 'SD'])
            plt.xlabel(r"$e = y - \^y$")
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title(
                'Error histogram - comparison between the methods (over test set)')
            plt.tight_layout()
            if save_hist:
                plt.savefig(imagepath_hist)
            plt.show()

        return e, y_hat_list

    def errorAnalysis_LLS(self, test_regressand, test_regressors):
        """  """
        e_LLS_te, y_hat_LLS_te = self.LLS_test(
            test_regressand, test_regressors)

        y_te = test_regressand.values
        X_te = test_regressors.values

        # Analysis for LLS
        # Training set
        E_tr_min_LLS = self.LLS_error_train.min()
        E_tr_max_LLS = self.LLS_error_train.max()
        E_tr_mu_LLS = self.LLS_error_train.mean()
        E_tr_sigma_LLS = self.LLS_error_train.std()
        E_tr_MSE_LLS = np.mean(self.LLS_error_train**2)
        # R^2 (coefficient of determination)
        R2_tr_LLS = 1 - E_tr_MSE_LLS/(np.std(self.regressand)**2)
        # Correlation coefficient
        c_tr_LLS = np.mean((self.regressand - self.regressand.mean())*(self.y_hat_LLS - self.y_hat_LLS.mean())
                           )/(self.regressand.std()*self.y_hat_LLS.std())

        # Test set
        E_te_min_LLS = e_LLS_te.min()
        E_te_max_LLS = e_LLS_te.max()
        E_te_mu_LLS = e_LLS_te.mean()
        E_te_sigma_LLS = e_LLS_te.std()
        E_te_MSE_LLS = np.mean(e_LLS_te**2)
        # R^2 (coefficient of determination)
        R2_te_LLS = 1 - E_te_MSE_LLS/(np.std(y_te)**2)
        # Correlation coefficient
        c_te_LLS = np.mean((y_te - y_te.mean())*(y_hat_LLS_te - y_hat_LLS_te.mean())
                           )/(y_te.std()*y_hat_LLS_te.std())

        rows = ['Training', 'Test']
        cols = ['min', 'max', 'mean', 'std', 'MSE', 'R^2', 'corr_coeff']
        p_LLS = np.array([
            [E_tr_min_LLS, E_tr_max_LLS, E_tr_mu_LLS,
                E_tr_sigma_LLS, E_tr_MSE_LLS, R2_tr_LLS, c_tr_LLS],
            [E_te_min_LLS, E_te_max_LLS, E_te_mu_LLS,
                E_te_sigma_LLS, E_te_MSE_LLS, R2_te_LLS, c_te_LLS]
        ])
        results_LLS = pd.DataFrame(p_LLS, columns=cols, index=rows)

        return results_LLS

    def errorAnalysis_SD(self, test_regressand, test_regressors):
        """  """

        e_SD_te, y_hat_SD_te = self.LLS_test(test_regressand, test_regressors)

        y_te = test_regressand.values
        X_te = test_regressors.values

        # Analysis for SD
        # Training set
        E_tr_min_SD = self.SD_error_train.min()
        E_tr_max_SD = self.SD_error_train.max()
        E_tr_mu_SD = self.SD_error_train.mean()
        E_tr_sigma_SD = self.SD_error_train.std()
        E_tr_MSE_SD = np.mean(self.SD_error_train**2)
        # R^2 (coefficient of determination)
        R2_tr_SD = 1 - E_tr_MSE_SD/(np.std(self.regressand)**2)
        # Correlation coefficient
        c_tr_SD = np.mean((self.regressand - self.regressand.mean())*(self.y_hat_SD - self.y_hat_SD.mean())
                          )/(self.regressand.std()*self.y_hat_SD.std())

        # Test set
        E_te_min_SD = e_SD_te.min()
        E_te_max_SD = e_SD_te.max()
        E_te_mu_SD = e_SD_te.mean()
        E_te_sigma_SD = e_SD_te.std()
        E_te_MSE_SD = np.mean(e_SD_te**2)
        # R^2 (coefficient of determination)
        R2_te_SD = 1 - E_te_MSE_SD/(np.std(y_te)**2)
        # Correlation coefficient
        c_te_SD = np.mean((y_te - y_te.mean())*(y_hat_SD_te - y_hat_SD_te.mean())
                          )/(y_te.std()*y_hat_SD_te.std())

        rows = ['Training', 'Test']
        cols = ['min', 'max', 'mean', 'std', 'MSE', 'R^2', 'corr_coeff']
        p_SD = np.array([
            [E_tr_min_SD, E_tr_max_SD, E_tr_mu_SD,
                E_tr_sigma_SD, E_tr_MSE_SD, R2_tr_SD, c_tr_SD],
            [E_te_min_SD, E_te_max_SD, E_te_mu_SD,
                E_te_sigma_SD, E_te_MSE_SD, R2_te_SD, c_te_SD]
        ])
        results_SD = pd.DataFrame(p_SD, columns=cols, index=rows)

        return results_SD

    def errorAnalysis(self, test_regressand, test_regressors):
        """  """

        res_LLS = self.errorAnalysis_LLS(test_regressand, test_regressors)
        res_SD = self.errorAnalysis_SD(test_regressand, test_regressors)

        features = res_LLS.columns
        # rows_LLS = res_LLS.index
        # rows_SD = res_SD.index

        p_LLS = res_LLS.values
        p_SD = res_SD.values
        p = np.concatenate((p_LLS, p_SD), axis=0)

        # Create a multi-index DataFrame, containing all data
        combinations = [
            ('LLS', 'Training'), ('LLS', 'Test'),
            ('SD', 'Training'), ('SD', 'Test')
        ]
        index_final = pd.MultiIndex.from_tuples(
            combinations, names=('Technique', 'Set'))

        result_full = pd.DataFrame(p, index=index_final, columns=features)

        return result_full

############################################################################################################

# Define function of euclidean distance in F-dimension


def dist_eval(element, train, dim=np.NaN):
    """
    dist_eval: evaluate the distance (euclidean sense) between the test element
    and each one of the elements of the training set
    ------------------------------------------------------------------------------
    - element: item whose distance needs to be computed
    - train: training set; each row is an element and the first 'dim' columns are 
      the features
    - dim: number of features considered in the distance (in this case )
    ------------------------------------------------------------------------------
    """
    if dim == np.NaN:
        # The number of features needs to be inferred
        dim = train.shape[1]

    distance_vect = np.empty((train.shape[0],))

    for ind2 in range(train.shape[0]):
        tmp_sum = sum(np.power(element - train[ind2, :], 2))
        distance_vect[ind2] = np.sqrt(tmp_sum)

    return distance_vect


class LocalLR():
    """ Local linear regression model """

    def __init__(self, regressand, regressors, N_closest):
        """  """
        # Check validity
        self.Np, self.Nf = regressors.values.shape
        if (regressand.values.shape[0] != self.Np):
            raise ValueError(
                "The dimensions of regressand and regressors are not coherent!\n")

        if (len(regressand.values.shape) > 1 and regressand.values.shape[1] > 1):
            raise ValueError("The regressand is not a 1D vector!\n")

        # Training set
        self.regressand = regressand.values
        self.regressors = regressors.values

        self.regressing_features = list(regressors.columns)
        # NOTE: y is a 'series' object, not a dataframe, since it consists of 1 dataframe column
        self.regressand_name = str(regressand.name)

        # Initialize solutions for Training Dataset (will be SD) - each w_hat will be one column of this matrix
        self.w_hat = np.zeros((self.Nf, self.Np))

        # Initialize approximated (training) regressands
        self.y_hat = np.zeros((self.Np, ))

        # The error on the training set will be
        self.SD_error_train = np.zeros((self.Np,))

        #################### Normalize values ############################################################
        # Define normalized values (on which the algorithm(s) will be performed)
        self.mean_regressors = self.regressors.mean(axis=0)
        self.stdev_regressors = self.regressors.std(axis=0)
        # WHAT TO DO WHEN STDEV=0? - set the standard deviation to a low value
        self.stdev_regressors[self.stdev_regressors == 0] = 1e-8

        if (not all(self.mean_regressors == np.zeros((self.Nf,))) or not all(self.stdev_regressors == np.ones((self.Nf,)))):
            # Normalize
            self.regressors_norm = (
                self.regressors - self.mean_regressors)/self.stdev_regressors
        else:
            self.regressors_norm = self.regressors

        #
        self.mean_regressand = self.regressand.mean(axis=0)
        self.stdev_regressand = self.regressand.std(axis=0)
        if self.stdev_regressand == 0:
            self.stdev_regressand = 1e-8

        if (self.mean_regressand != 0 or self.stdev_regressand != 1):
            # Normalize
            self.regressand_norm = (
                self.regressand - self.mean_regressand)/self.stdev_regressand
        else:
            self.regressand_norm = self.regressand

        ##################################################################################################

        # Number of closest
        self.N = N_closest

    def solve(self, plot_y=False, save_y=False, imagepath_y="./img/LOCAL_training-y_vs_y_hat.png", plot_hist=False, save_hist=False, imagepath_hist='./img/LOCAL-train_err_hist.png'):
        """ 
        Solution of the Local Linear Regression given N closest neighbors for the training 
        dataset
        -----------------------------------------------------------------------------------
        Approach:
        - Iterate over each patient
        - Find N closest
        - Create new regressand and regressors
            - Call new LinearRegression object
        - Find w_hat, store it in a new column of self.w_hat (use SD - as required)
        - Find y_hat_norm and store it in a new column of self.y_hat_norm
        -----------------------------------------------------------------------------------
        """
        Np = self.Np        # Number f patients in this set
        Nf = self.Nf        # Number of features
        N = self.N          # Number of considered neighbors

        # Isolate the regressors matrix and the regressand (training dataset)
        X = np.copy(self.regressors_norm)
        y = np.copy(self.regressand_norm)

        for i in range(Np):
            # Isolate current patient
            x_curr = X[i, :]
            y_curr = y[i]

            # Evaluate the distance of the current patient wrt each patient in the training set
            distances = dist_eval(x_curr, X, dim=Nf)

            # Find elements with minimum distance (i.e., indices of the N closest elements)
            sorted_vect = np.argsort(distances)

            # Find the N closest elements
            X_closest = np.empty((N, Nf))
            y_closest = np.empty((N,))
            for ind1 in range(N):
                # Fill X_closest & y_closest (corresponding patients)
                X_closest[ind1, :] = np.copy(X[sorted_vect[ind1], :])
                y_closest[ind1] = np.copy(y[sorted_vect[ind1]])

            # Create new LinearRegression object with X_closest and y_closest
            # BUT: need dataframes
            X_closest_df = pd.DataFrame(
                X_closest, columns=self.regressing_features)
            y_closest_df = pd.Series(
                y_closest, name=self.regressand_name)

            # TODO: Create new LinearRegression object - NO NEED TO NORMALIZE
            LR_closest = LinearRegression(
                y_closest_df, X_closest_df, normalize=False)
            # Update w_hat_SD
            LR_closest.solve_SteepestDescent()

            # Get weights for Local LR
            w_hat_SD_curr = LR_closest.w_hat_SD
            # Store it in column i of self.w_hat
            self.w_hat[:, i] = np.copy(w_hat_SD_curr)

            # Get y_hat of the current patient (x_curr)
            y_hat_curr = np.reshape(
                x_curr, (1, len(x_curr)))@np.reshape(w_hat_SD_curr, (len(w_hat_SD_curr), 1))
            # Store this result in element i of the vector self.y_hat
            # Need to de-normalize first
            self.y_hat[i] = np.copy(
                y_hat_curr*self.stdev_regressand + self.mean_regressand)

            # Find error on the training set
            error_curr = self.y_hat[i] - self.regressand[i]
            # Place it in position i
            self.SD_error_train[i] = np.copy(error_curr)

        # Plot y vs y_hat (copy before)
        if plot_y:
            plt.figure(figsize=(6, 4))
            plt.plot(self.regressand, self.y_hat, '.')   # Place dots
            v = plt.axis()
            # Plot 45deg diagonal line
            plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
            plt.xlabel(r'$y$')
            plt.ylabel(r'$\^y$', rotation=0)
            plt.grid()
            plt.title("Local Linear Regression - Training validation set")
            plt.tight_layout()
            if save_y:
                plt.savefig(imagepath_y)
            plt.show()

        # Plot error histogram
        if plot_hist:
            e = self.SD_error_train

            plt.figure(figsize=(6, 4))
            plt.hist(e, bins=50, density=True, histtype='bar',
                     label='training')
            plt.xlabel(r"$e = y - \^y$")
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title('Local Linear Regression - Training error histogram')
            plt.tight_layout()
            if save_hist:
                plt.savefig(imagepath_hist)
            plt.show()

        return self.w_hat, self.y_hat, self.SD_error_train

    def test(self, test_regressand, test_regressors, plot_y=False, save_y=False, imagepath_y="./img/LOCAL_test-y_vs_y_hat.png", plot_hist=False, save_hist=False, imagepath_hist='./img/LOCAL-test_err_hist.png'):
        """ Test local regression model """
        # Find K nearest
        Np_test = test_regressand.shape[0]      # n. patients in the test set
        Nf = self.Nf                            # number of features
        N = self.N

        # Check dimensions
        if test_regressors.shape[0] != Np_test:
            raise ValueError(
                "Error! The number of patients in the test regressors matrix is different from the one in the regressand!")

        if test_regressors.values.shape[1] != Nf:
            raise ValueError(
                "Error! The number of features in the training set is not the same as in the test set!")

        # Isolate ndarrays of test set
        y_test = test_regressand.values
        X_test = test_regressors.values

        # Normalize (wrt test set parameters)
        X_test_norm = (X_test - self.mean_regressors)/self.stdev_regressors
        y_test_norm = (y_test - self.mean_regressand)/self.stdev_regressand

        # Take training set
        X_tr = self.regressors
        y_tr = self.regressand

        w_hat_te = np.zeros((Nf, Np_test))      # Regression result
        y_hat_te = np.zeros((Np_test, ))        # Approximated y (test set)
        err_test = np.zeros((Np_test, ))

        # Iterate on all patients in the test set:
        for i in range(X_test_norm.shape[0]):
            x_current = X_test_norm[i, :]
            y_current = y_test_norm[i]

            distances = dist_eval(x_current, X_tr, dim=Nf)

            # Find elements with minimum distance (i.e., indices of the N closest elements)
            sorted_vect = np.argsort(distances)

            # Find the N closest elements
            X_closest_te = np.empty((N, Nf))
            y_closest_te = np.empty((N,))

            for ind1 in range(N):
                # Fill X_closest with closest elements in the training set
                X_closest_te[ind1, :] = np.copy(X_tr[sorted_vect[ind1], :])
                y_closest_te[ind1] = np.copy(y_tr[sorted_vect[ind1]])

            # Create new LinearRegression object with X_closest and y_closest
            # BUT: need dataframes
            X_closest_te_df = pd.DataFrame(
                X_closest_te, columns=self.regressing_features)
            y_closest_te_df = pd.Series(
                y_closest_te, name=self.regressand_name)

            LR_closest_test = LinearRegression(
                y_closest_te_df, X_closest_te_df, normalize=False)
            # Update w_hat_SD
            LR_closest_test.solve_SteepestDescent()

            # Get weights for Local LR
            w_hat_te_curr = LR_closest_test.w_hat_SD
            # Store it in column i of w_hat_te
            w_hat_te[:, i] = np.copy(w_hat_te_curr)

            # Get y_hat of the current patient (x_curr)
            y_hat_te_curr = np.reshape(
                x_current, (1, len(x_current))) @ np.reshape(w_hat_te_curr, (len(w_hat_te_curr), 1))

            # Store this result in element i of the vector self.y_hat
            # NOTE: after denormalizing
            y_hat_te[i] = np.copy(
                y_hat_te_curr*self.stdev_regressand + self.mean_regressand)

            # Error - on de-normalized values
            err_test[i] = y_test[i] - y_hat_te[i]

        # Plot y vs. y_hat
        if plot_y:
            plt.figure(figsize=(6, 4))
            plt.plot(y_test, y_hat_te, '.')   # Place dots
            v = plt.axis()
            # Plot 45deg diagonal line
            plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
            plt.xlabel(r'$y$')
            plt.ylabel(r'$\^y$', rotation=0)
            plt.grid()
            plt.title("Local Linear Regression - Test set")
            plt.tight_layout()
            if save_y:
                plt.savefig(imagepath_y)
            plt.show()

        # Plot error histogram
        if plot_hist:
            e = err_test

            plt.figure(figsize=(6, 4))
            plt.hist(e, bins=50, density=True, histtype='bar',
                     label='training')
            plt.xlabel(r"$e = y - \^y$")
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title('Local Linear Regression - Test error histogram')
            plt.tight_layout()
            if save_hist:
                plt.savefig(imagepath_hist)
            plt.show()

        return err_test, y_hat_te, w_hat_te

    # TODO:
    # Define a function for comparing errors on test and train sets (histogram) +
    # Creating the output DataFrame
    def errorAnalysis(self, test_regressand, test_regressors, plot_hist=False, save_hist=False, imagepath_hist='./img/LOCAL_error-hist_train-vs-test.png'):
        if (self.w_hat == 0).all():
            self.solve()

        err_test, y_hat_te = self.test(test_regressand, test_regressors)

        err = [self.SD_error_train, err_test]

        if plot_hist:
            plt.figure(figsize=(6, 4))
            plt.hist(err, bins=50, density=True, histtype='bar',
                     label=['Training set', 'Test set'])
            plt.xlabel(r"$e = y - \^y$")
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title(
                'Local Linear Regression error histogram - comparison between the training and test set')
            plt.tight_layout()
            if save_hist:
                plt.savefig(imagepath_hist)
            plt.show()

        # Produce DF
        y_te = test_regressand.values

        # Analysis for SD
        # Training set
        E_tr_min = self.SD_error_train.min()
        E_tr_max = self.SD_error_train.max()
        E_tr_mu = self.SD_error_train.mean()
        E_tr_sigma = self.SD_error_train.std()
        E_tr_MSE = np.mean(self.SD_error_train**2)
        # R^2 (coefficient of determination)
        R2_tr = 1 - E_tr_MSE/(np.std(self.regressand)**2)
        # Correlation coefficient
        c_tr = np.mean((self.regressand - self.regressand.mean())*(self.y_hat - self.y_hat.mean())
                       )/(self.regressand.std()*self.y_hat.std())

        # Test set
        E_te_min = err_test.min()
        E_te_max = err_test.max()
        E_te_mu = err_test.mean()
        E_te_sigma = err_test.std()
        E_te_MSE = np.mean(err_test**2)
        # R^2 (coefficient of determination)
        R2_te = 1 - E_te_MSE/(np.std(y_te)**2)
        # Correlation coefficient
        c_te = np.mean((y_te - y_te.mean())*(y_hat_te - y_hat_te.mean())
                       )/(y_te.std()*y_hat_te.std())

        rows = ['Training', 'Test']
        cols = ['min', 'max', 'mean', 'std', 'MSE', 'R^2', 'corr_coeff']
        p_SD = np.array([
            [E_tr_min, E_tr_max, E_tr_mu,
                E_tr_sigma, E_tr_MSE, R2_tr, c_tr],
            [E_te_min, E_te_max, E_te_mu,
                E_te_sigma, E_te_MSE, R2_te, c_te]
        ])
        results_local = pd.DataFrame(p_SD, columns=cols, index=rows)

        return results_local


if __name__ == '__main__':
    pass
