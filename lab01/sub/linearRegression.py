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
    - train_regressand: Pandas DataFrame containing the training regressand
    - train_regressors: Pandas Dataframe containing the training regressors
    - test_regressand: Pandas DataFrame containing the test regressand (NOT COMPULSORY)
    - test_regressors: Pandas Dataframe containing the test regressors (NOT COMPULSORY)
    - normalize: (default True) specify whether we need to normalize the input 
      (training) dataset before computing linear regression
    -----------------------------------------------------------------------------------
    Attributes:
    - Np_tr: number rows in the training regressors matrix
    - Np_te: number rows in the test regressors matrix
    - Nf: number of regressing features
    - train_regressand: regressand vector - Np elements (np.Ndarray)
    - train_regressors: regressors matrix - Np x Nf (np.Ndarray)
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
    - test_defined: True if the test set was provided
    -----------------------------------------------------------------------------------
    """

    def __init__(self, train_regressand, train_regressors, test_regressand = pd.DataFrame({'A' : []}), test_regressors = pd.DataFrame({'A' : []}), normalize=True):
        # Check validity
        self.Np_tr, self.Nf = train_regressors.values.shape
        if (test_regressand.empty or test_regressors.empty):
            # Allow for calling this class without providing test set
            self.test_defined = False
        else: self.test_defined = True

        self.Np_te = test_regressors.values.shape[0]

        # Same number of features between train and test
        if self.test_defined and (train_regressors.values.shape[1] != self.Nf):
            raise ValueError("The training and test regressing features are not the same!")

        # Same rows in regressand and regressors (training)
        if (train_regressand.values.shape[0] != self.Np_tr):
            raise ValueError(
                "The dimensions of training regressand and regressors are not coherent!\n")
        
        # Same rows in regressand and regressors (test)
        if self.test_defined and (test_regressand.values.shape[0] != self.Np_te):
            raise ValueError(
                "The dimensions of test regressand and regressors are not coherent!\n")

        # Check training regressand
        if (len(train_regressand.values.shape) > 1 and train_regressand.values.shape[1] > 1):
            raise ValueError("The training regressand is not a 1D vector!\n")

        # Check test regressand
        if self.test_defined and (len(test_regressand.values.shape) > 1 and test_regressand.values.shape[1] > 1):
            raise ValueError("The test regressand is not a 1D vector!\n")

        ##################### Data set ##############################################

        self.train_regressand = train_regressand.values
        self.train_regressors = train_regressors.values

        self.test_regressand = test_regressand.values if (self.test_defined) else np.zeros((self.Np_tr,), dtype=float)
        self.test_regressors = test_regressors.values if (self.test_defined) else np.zeros((self.Np_tr, self.Nf), dtype=float)

        # Feature names
        self.regressing_features = list(train_regressors.columns)
        # NOTE: y is a 'Series' object, not a DataFrame, since it only contains 1 column
        self.regressand_name = str(train_regressand.name)

        # Initialize solutions
        self.w_hat_LLS = np.zeros((self.Nf,), dtype=float)
        self.w_hat_SD = np.zeros((self.Nf,), dtype=float)

        # Initialize approximated train regressands
        self.y_hat_tr_LLS = np.zeros((self.Np_tr,), dtype=float)
        self.y_hat_tr_SD = np.zeros((self.Np_tr,), dtype=float)

        # Initialize approximated test regressands
        self.y_hat_te_LLS = np.zeros((self.Np_te,), dtype=float)
        self.y_hat_te_SD = np.zeros((self.Np_te,), dtype=float)

        # Initialize normalized approximated train regressands
        self.y_hat_tr_LLS_norm = np.zeros((self.Np_tr,), dtype=float)
        self.y_hat_tr_SD_norm = np.zeros((self.Np_tr,), dtype=float)

        # Initialize normalized approximated test regressands
        self.y_hat_te_LLS_norm = np.zeros((self.Np_te,), dtype=float)
        self.y_hat_te_SD_norm = np.zeros((self.Np_te,), dtype=float)

        if normalize:
            # Define normalized values (on which the algorithm(s) will be performed)
            self.mean_regressors = self.train_regressors.mean(axis=0)
            self.stdev_regressors = self.train_regressors.std(axis=0)

            if (0 in self.stdev_regressors):
                print("One of the standard deviations is 0")
                print(self.regressing_features)
                print(self.stdev_regressors)

            # WHAT TO DO WHEN STDEV=0? - set the standard deviation to a low value
            self.stdev_regressors[self.stdev_regressors == 0] = 1e-8

            # Normalize training set 
            if (not all(self.mean_regressors == np.zeros((self.Nf,))) or not all(self.stdev_regressors == np.ones((self.Nf,)))):
                # Normalize
                self.train_regressors_norm = (
                    self.train_regressors - self.mean_regressors)/self.stdev_regressors
                
                self.test_regressors_norm = (self.test_regressors - self.mean_regressors)/self.stdev_regressors

            else:
                # Do not normalize
                self.train_regressors_norm = self.train_regressors
                self.test_regressors_norm = self.test_regressors

            self.mean_regressand = self.train_regressand.mean(axis=0)
            self.stdev_regressand = self.train_regressand.std(axis=0)
            if self.stdev_regressand == 0:  # Case stdev == 0
                self.stdev_regressand = 1e-8

            if (self.mean_regressand != 0 or self.stdev_regressand != 1) and (normalize):
                # Normalize
                self.train_regressand_norm = (
                    self.train_regressand - self.mean_regressand)/self.stdev_regressand

                self.test_regressand_norm = (
                    self.test_regressand - self.mean_regressand)/self.stdev_regressand

            else:
                self.train_regressand_norm = self.train_regressand
                self.test_regressand_norm = self.test_regressand

        else:
            # No normalization - set all means to 0 and stdev to 1
            self.mean_regressors = np.zeros((1, len(self.regressing_features)), dtype=float)
            self.stdev_regressors = np.ones((1, len(self.regressing_features)), dtype=float)
            self.train_regressors_norm = self.train_regressors
            self.test_regressors_norm = self.test_regressors

            self.mean_regressand = 0
            self.stdev_regressand = 1
            self.train_regressand_norm = self.train_regressand
            self.test_regressand_norm = self.test_regressand

            # NOTE: regressors_norm and regressand_norm will not necessarily be normalized in this case

        # Init. error vectors
        self.LLS_error_train = np.zeros((self.Np_tr,), dtype=float)
        self.SD_error_train = np.zeros((self.Np_tr,), dtype=float)

        self.LLS_error_test = np.zeros((self.Np_te,), dtype=float)
        self.SD_error_test = np.zeros((self.Np_te,), dtype=float)

    def solve_LLS(self, plot_w=False, save_w=False, imagepath_w="./img/01_LLS-w_hat.png", plot_y=False, save_y=False, imagepath_y="./img/02_LLS-y_vs_y_hat.png"):
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
        X_tr_norm = self.train_regressors_norm
        y_tr_norm = self.train_regressand_norm
        #self.w_hat_LLS = np.linalg.inv(
        #    X_tr_norm.T@X_tr_norm)@(X_tr_norm.T@y_tr_norm)

        self.w_hat_LLS = mymin.SolveLLS(y_tr_norm, X_tr_norm).run()

        self.y_hat_tr_LLS_norm = X_tr_norm@(self.w_hat_LLS)
        self.y_hat_tr_LLS = self.stdev_regressand * \
            (self.y_hat_tr_LLS_norm) + self.mean_regressand

        self.LLS_error_train = self.train_regressand - self.y_hat_tr_LLS

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
            plt.plot(self.train_regressand, self.y_hat_tr_LLS, '.')   # Place dots
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

    def solve_SteepestDescent(self, stoppingCondition='iterations', Nit=100, plot_w=False, save_w=False, imagepath_w="./img/03_SD-w_hat.png", plot_y=False, save_y=False, imagepath_y="./img/04_SD-y_vs_y_hat.png"):
        """
        Solution of the Linear Regression by means of the Steepest Descent method.
        This function fills the attribute w_hat_SD.
        -----------------------------------------------------------------------------------
        Optional parameters: 
        - stoppingCondition: (default 'iterations') decides which stopping condition to use
          (can be: 'iterations' and 'epsilon')
        - Nit: (default 50) number of iterations (stopping condition) for the Steepest Descent algorithm
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
        X_tr_norm = self.train_regressors_norm
        y_tr_norm = self.train_regressand_norm
        SD_problem = mymin.SteepestDescent(
            y_tr_norm, X_tr_norm)
        self.w_hat_SD = SD_problem.run(
            stoppingCondition=stoppingCondition, Nit=Nit)

        self.y_hat_tr_SD_norm = X_tr_norm@(self.w_hat_SD)
        self.y_hat_tr_SD = self.stdev_regressand * \
            (self.y_hat_tr_SD_norm) + self.mean_regressand

        self.SD_error_train = self.train_regressand - self.y_hat_tr_SD

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
            plt.plot(self.train_regressand, self.y_hat_tr_SD, '.')   # Place dots
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

    def plot_w(self, save_png=False, imagepath="./img/05_w_hat_comparison.png"):
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
        plt.figure(figsize=(9, 4))
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

    def LLS_test(self, plot_hist=False, save_hist=False, imagepath_hist='./img/06_LLS-err_hist.png', plot_y=False, save_y=False, imagepath_y='./img/07_LLS_y_test_vs_y_hat_test.png'):
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
        # Check for availability of the test set
        if not self.test_defined:
            raise RuntimeError("The test set was not defined!")

        # Check w_hat_LLS already computed
        if (all(self.w_hat_LLS == np.zeros((self.Nf,)))):
            self.solveLLS()

        ## Test set ###################################################
    
        # Normalize
        X_test_norm = self.test_regressors_norm

        # Obtain approximated regressand
        self.y_hat_te_LLS_norm = X_test_norm@self.w_hat_LLS
        # De-normalize
        self.y_hat_te_LLS = self.y_hat_te_LLS_norm*self.stdev_regressand + self.mean_regressand

        # Error
        self.LLS_error_test = self.test_regressand - self.y_hat_te_LLS

        if plot_hist:
            e = [self.LLS_error_train, self.LLS_error_test]

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
            plt.plot(self.test_regressand, self.y_hat_te_LLS, '.')   # Place dots
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

        return self.LLS_error_test, self.y_hat_te_LLS

    def SD_test(self, plot_hist=False, save_hist=False, imagepath_hist='./img/08_SD-err_hist.png', plot_y=False, save_y=False, imagepath_y='./img/09_SD_y_test_vs_y_hat_test.png'):
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
        # Check for availability of the test set
        if not self.test_defined:
            raise RuntimeError("The test set was not defined!")

        # Check w_hat_SD already computed
        if (all(self.w_hat_SD == np.zeros((self.Nf,)))):
            self.solveSD()

        ## Test set ###################################################

        # Normalize
        X_test_norm = self.test_regressors_norm

        # Obtain approximated regressand
        self.y_hat_te_SD_norm = X_test_norm@self.w_hat_SD
        # De-normalize
        self.y_hat_te_SD = self.y_hat_te_SD_norm*self.stdev_regressand + self.mean_regressand

        # Error
        self.SD_error_test = self.test_regressand - self.y_hat_te_SD

        if plot_hist:
            e = [self.SD_error_train, self.SD_error_test]

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
            plt.plot(self.test_regressand, self.y_hat_te_SD, '.')   # Place dots
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

        return self.SD_error_test, self.y_hat_te_SD

    def test(self, plot_hist=False, save_hist=False, imagepath_hist='./img/10_err_hist_compare.png'):
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
        # Check for availability of the test set
        if not self.test_defined:
            raise RuntimeError("The test set was not defined!")
        
        # Call both LLS_test and SD_test and store the results (absolute errors) in order
        # to make the comparison between the two methods
    
        # Check whether the solution was already evaluated or not
        if (all(self.y_hat_te_LLS == np.zeros((self.Np_te,)))):
            self.LLS_test()

        if (all(self.y_hat_te_SD == np.zeros((self.Np_te,)))):
            self.SD_test()

        e = [self.LLS_error_test, self.SD_error_test]
        y_hat_list = [self.y_hat_te_LLS, self.y_hat_te_SD]

        if plot_hist:
            plt.figure(figsize=(8, 5))
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

    def errorAnalysis_LLS(self):
        """
        Produce a DataFrame containing figures of merit highlighting the performance of Linear Regression with LLS
        on both the Test set and the Training set
        """
        # Check for availability of the test set
        if not self.test_defined:
            raise RuntimeError("The test set was not defined!")

        if (all(self.y_hat_tr_LLS == np.zeros((self.Np_tr,)))):
            self.solve_LLS()

        # Check whether the solution was already evaluated or not
        if (all(self.y_hat_te_LLS == np.zeros((self.Np_te,)))):
            self.LLS_test()

        # Analysis for LLS
        # Training set
        E_tr_min_LLS = self.LLS_error_train.min()
        E_tr_max_LLS = self.LLS_error_train.max()
        E_tr_mu_LLS = self.LLS_error_train.mean()
        E_tr_sigma_LLS = self.LLS_error_train.std()
        E_tr_MSE_LLS = np.mean(self.LLS_error_train**2)
        # R^2 (coefficient of determination)
        R2_tr_LLS = 1 - E_tr_MSE_LLS/(np.std(self.train_regressand)**2)
        # Correlation coefficient
        c_tr_LLS = np.mean((self.train_regressand - self.mean_regressand)*(self.y_hat_tr_LLS - self.y_hat_tr_LLS.mean())
                           )/(self.stdev_regressand*self.y_hat_tr_LLS.std())

        # Test set
        E_te_min_LLS = self.LLS_error_test.min()
        E_te_max_LLS = self.LLS_error_test.max()
        E_te_mu_LLS = self.LLS_error_test.mean()
        E_te_sigma_LLS = self.LLS_error_test.std()
        E_te_MSE_LLS = np.mean(self.LLS_error_test**2)
        # R^2 (coefficient of determination)
        R2_te_LLS = 1 - E_te_MSE_LLS/(np.std(self.test_regressand)**2)
        # Correlation coefficient
        c_te_LLS = np.mean((self.test_regressand - self.test_regressand.mean())*(self.y_hat_te_LLS - self.y_hat_te_LLS.mean())
                           )/(self.test_regressand.std()*self.y_hat_te_LLS.std())

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

    def errorAnalysis_SD(self):
        """
        Produce a DataFrame containing figures of merit highlighting the performance of Linear Regression with Steepest Descent
        on both the Test set and the Training set
        """
        # Check for availability of the test set
        if not self.test_defined:
            raise RuntimeError("The test set was not defined!")

        if (all(self.y_hat_tr_SD == np.zeros((self.Np_tr,)))):
            self.solve_SteepestDescent()

        if (all(self.y_hat_te_SD == np.zeros((self.Np_te,)))):
            self.SD_test()

        # Analysis for SD
        # Training set
        E_tr_min_SD = self.SD_error_train.min()
        E_tr_max_SD = self.SD_error_train.max()
        E_tr_mu_SD = self.SD_error_train.mean()
        E_tr_sigma_SD = self.SD_error_train.std()
        E_tr_MSE_SD = np.mean(self.SD_error_train**2)
        # R^2 (coefficient of determination)
        R2_tr_SD = 1 - E_tr_MSE_SD/(np.std(self.train_regressand)**2)
        # Correlation coefficient
        c_tr_SD = np.mean((self.train_regressand - self.mean_regressand)*(self.y_hat_tr_SD - self.y_hat_tr_SD.mean())
                          )/(self.stdev_regressand*self.y_hat_tr_SD.std())

        # Test set
        E_te_min_SD = self.SD_error_test.min()
        E_te_max_SD = self.SD_error_test.max()
        E_te_mu_SD = self.SD_error_test.mean()
        E_te_sigma_SD = self.SD_error_test.std()
        E_te_MSE_SD = np.mean(self.SD_error_test**2)
        # R^2 (coefficient of determination)
        R2_te_SD = 1 - E_te_MSE_SD/(np.std(self.test_regressand)**2)
        # Correlation coefficient
        c_te_SD = np.mean((self.test_regressand - self.test_regressand.mean())*(self.y_hat_te_SD - self.y_hat_te_SD.mean())
                          )/(self.test_regressand.std()*self.y_hat_te_SD.std())

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

    def errorAnalysis(self):
        """
        Produce a DataFrame containing figures of merit highlighting the performance of Linear regression (with either LLS
        or Steepest Descent) on both the Test set and the Training set
        """
        # Check for availability of the test set
        if not self.test_defined:
            raise RuntimeError("The test set was not defined!")

        res_LLS = self.errorAnalysis_LLS()
        res_SD = self.errorAnalysis_SD()

        features = res_LLS.columns

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


def dist_eval(element, train):
    """
    dist_eval: evaluate the distance (euclidean sense) between the test element
    and each one of the elements of the training set
    ------------------------------------------------------------------------------
    - element: item whose distance needs to be computed
    - train: training set; each row is an element and the first 'dim' columns are 
      the features
    ------------------------------------------------------------------------------
    """
    # Check same n. of features
    if element.shape[0] != train.shape[1]:
        raise ValueError('Error! The number of features of the element is not the same as the one of the training set!')

    distance_vect = np.empty((train.shape[0],))

    for ind2 in range(train.shape[0]):
        tmp_sum = sum(np.power(element - train[ind2, :], 2))
        distance_vect[ind2] = np.sqrt(tmp_sum)

    return distance_vect


class LocalLR():
    """ 
    Local linear regression model 
    ------------------------------------------------------------------------------
    Attributes:
    - Np_tr
    - Np_te
    - Nf
    - train_regressand
    - train_regressors
    - test_regressand
    - test_regressors
    - regressing_features
    - regressand_name
    - w_hat_tr
    - w_hat_te
    - y_hat_tr
    - y_hat_te
    - err_train
    - mean_regressors
    - stdev_regressors
    - mean_regressand
    - stdev_regressand
    - train_regressors_norm
    - train_regressand_norm
    - test_regressors_norm
    - test_regressand_norm
    - N
    ------------------------------------------------------------------------------
    """

    def __init__(self, train_regressand, train_regressors, test_regressand, test_regressors, N_closest):
        # Check validity
        self.Np_tr, self.Nf = train_regressors.values.shape
        self.Np_te = test_regressors.values.shape[0]

        # Same number of features between train and test
        if (train_regressors.values.shape[1] != self.Nf):
            raise ValueError("The training and test regressing features are not the same!")

        # Same rows in regressand and regressors (training)
        if (train_regressand.values.shape[0] != self.Np_tr):
            raise ValueError(
                "The dimensions of training regressand and regressors are not coherent!\n")
        
        # Same rows in regressand and regressors (test)
        if (test_regressand.values.shape[0] != self.Np_te):
            raise ValueError(
                "The dimensions of test regressand and regressors are not coherent!\n")

        # Check training regressand
        if (len(train_regressand.values.shape) > 1 and train_regressand.values.shape[1] > 1):
            raise ValueError("The training regressand is not a 1D vector!\n")

        # Check test regressand
        if (len(test_regressand.values.shape) > 1 and test_regressand.values.shape[1] > 1):
            raise ValueError("The test regressand is not a 1D vector!\n")

        #################### Data Set #############################################################
        self.train_regressand = train_regressand.values
        self.train_regressors = train_regressors.values

        self.test_regressand = test_regressand.values
        self.test_regressors = test_regressors.values

        # Feature names
        self.regressing_features = list(train_regressors.columns)
        # NOTE: y is a 'Series' object, not a DataFrame, since it only contains 1 column
        self.regressand_name = str(train_regressand.name)

        # Initialize solutions for Training Dataset (will be SD) - each w_hat will be one column of this matrix
        self.w_hat_tr = np.zeros((self.Nf, self.Np_tr), dtype=float)
        # Initialize solutions for Test Dataset (will be SD) - each w_hat will be one column of this matrix
        self.w_hat_te = np.zeros((self.Nf, self.Np_te), dtype=float)

        # Initialize approximated (training) regressands
        self.y_hat_tr = np.zeros((self.Np_tr, ), dtype=float)
        # Initialize approximated (test) regressands
        self.y_hat_te = np.zeros((self.Np_te, ), dtype=float)

        # The error on the training set will be
        self.err_train = np.zeros((self.Np_tr,), dtype=float)
        # The error on the test set will be
        self.err_test = np.zeros((self.Np_te,), dtype=float)

        #################### Normalize values ############################################################
        # Define normalized values (on which the algorithm(s) will be performed)
        self.mean_regressors = self.train_regressors.mean(axis=0)
        self.stdev_regressors = self.train_regressors.std(axis=0)
        # WHAT TO DO WHEN STDEV=0? - set the standard deviation to a low value
        self.stdev_regressors[self.stdev_regressors == 0] = 1e-8

        if (not all(self.mean_regressors == np.zeros((self.Nf,))) or not all(self.stdev_regressors == np.ones((self.Nf,)))):
            # Normalize
            self.train_regressors_norm = (
                self.train_regressors - self.mean_regressors)/self.stdev_regressors

            self.test_regressors_norm = (
                self.test_regressors - self.mean_regressors)/self.stdev_regressors
        else:
            self.train_regressors_norm = self.train_regressors
            self.test_regressors_norm = self.test_regressors

        #
        self.mean_regressand = self.train_regressand.mean(axis=0)
        self.stdev_regressand = self.train_regressand.std(axis=0)
        if self.stdev_regressand == 0:
            self.stdev_regressand = 1e-8

        if (self.mean_regressand != 0 or self.stdev_regressand != 1):
            # Normalize
            self.train_regressand_norm = (
                self.train_regressand - self.mean_regressand)/self.stdev_regressand

            self.test_regressand_norm = (
                self.test_regressand - self.mean_regressand)/self.stdev_regressand
        else:
            self.train_regressand_norm = self.train_regressand
            self.test_regressand_norm = self.test_regressand

        ##################################################################################################

        # Number of closest
        self.N = N_closest

    def solve(self, plot_y=False, save_y=False, imagepath_y=f"./img/11_LOCAL_training-y_vs_y_hat.png", plot_hist=False, save_hist=False, imagepath_hist='./img/12_LOCAL-training_err_hist.png'):
        """ 
        Test local regression model 
        ------------------------------------------------------------------------------
        Solution of the Local Linear Regression given N closest neighbors for the 
        Training dataset.
        
        Approach:
        - Iterate over each patient in the NORMALIZED training set
        - Find N closest (in the NORMALIZED regressors)
        - Create new regressand and regressors
            - Call new LinearRegression object
        - Find w_hat, store it in a new column of self.w_hat (use SD - as required)
        - Find y_hat_norm and store it in a new column of self.y_hat_norm
        ------------------------------------------------------------------------------
        Input parameters:
        - plot_y (default False): used to plot the comparison y_hat vs. y
        - save_y (default False): used to save the plot of comparison for y
        - impagepath_y (default "./img/LOCAL_training-y_vs_y_hat.png"): path specifying 
          where to store the y plot
        - plot_hist (default False): used to plot the histogram of the error
        - save_hist (default False)
        - imagepath_hist (default './img/LOCAL-training_err_hist.png')
        ------------------------------------------------------------------------------
        Returned variables:
        - self.err_train: vector containing the error done by the local regression 
          approximation (length equal to the regressand)
        - self.y_hat: (de-normalized) vector of the approximated regressand
        - self.w_hat: matrix whose columns are the weights vectors associated with each 
          row of the regressor matrix
        ------------------------------------------------------------------------------
        """

        Np = self.Np_tr        # Number of patients in this set
        Nf = self.Nf        # Number of features
        N = self.N          # Number of considered neighbors

        # Isolate the NORMALIZED regressors matrix and the regressand (training dataset)
        X_tr = np.copy(self.train_regressors_norm)
        y = np.copy(self.train_regressand_norm)

        for i in range(Np):
            # Isolate current patient
            x_curr = X_tr[i, :]
            y_curr = y[i]

            # Evaluate the distance of the current patient wrt each patient in the training set
            distances = dist_eval(x_curr, X_tr)

            # Find elements with minimum distance (i.e., indices of the N closest elements)
            sorted_vect = np.argsort(distances)

            # Find the N closest elements
            X_closest = np.empty((N, Nf))
            y_closest = np.empty((N,))
            for ind1 in range(N):
                # Fill X_closest & y_closest (corresponding patients)
                X_closest[ind1, :] = np.copy(X_tr[sorted_vect[ind1], :])
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
            self.w_hat_tr[:, i] = np.copy(w_hat_SD_curr)

            # Get y_hat of the current patient (x_curr)
            y_hat_curr = np.reshape(
                x_curr, (1, len(x_curr)))@np.reshape(w_hat_SD_curr, (len(w_hat_SD_curr), 1))
            # Store this result in element i of the vector self.y_hat
            # Need to de-normalize first
            self.y_hat_tr[i] = np.copy(
                y_hat_curr*self.stdev_regressand + self.mean_regressand)

            # Find error on the training set
            error_curr = self.y_hat_tr[i] - self.train_regressand[i]
            # Place it in position i
            self.err_train[i] = np.copy(error_curr)

        # Plot y vs y_hat (copy before)
        if plot_y:
            plt.figure(figsize=(6, 4))
            plt.plot(self.train_regressand, self.y_hat_tr, '.')   # Place dots
            v = plt.axis()
            # Plot 45deg diagonal line
            plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
            plt.xlabel(r'$y$')
            plt.ylabel(r'$\^y$', rotation=0)
            plt.grid()
            plt.title(f"Local Linear Regression - Training validation set - N={self.N}")
            plt.tight_layout()
            if save_y:
                plt.savefig(imagepath_y)
            plt.show()

        # Plot error histogram
        if plot_hist:
            e = self.err_train

            plt.figure(figsize=(6, 4))
            plt.hist(e, bins=50, density=True, histtype='bar',
                     label='training')
            plt.xlabel(r"$e = y - \^y$")
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title(f'Local Linear Regression - Training error histogram - N={self.N}')
            plt.tight_layout()
            if save_hist:
                plt.savefig(imagepath_hist)
            plt.show()

        return self.err_train, self.y_hat_tr, self.w_hat_tr

    def test(self, plot_y=False, save_y=False, imagepath_y="./img/13_LOCAL_test-y_vs_y_hat.png", plot_hist=False, save_hist=False, imagepath_hist='./img/14_LOCAL-test_err_hist.png'):
        """ 
        Test local regression model 
        ------------------------------------------------------------------------------
        Input parameters:
        - test_regressand: DataFrame
        - Test_regressors: DataFrame
        - plot_y (default False): used to plot the comparison y_hat vs. y
        - save_y (default False): used to save the plot of comparison for y
        - impagepath_y (default "./img/LOCAL_test-y_vs_y_hat.png"): path specifying 
          where to store the y plot
        - plot_hist (default False): used to plot the histogram of the error
        - save_hist (default False)
        - imagepath_hist (default './img/LOCAL-test_err_hist.png')
        ------------------------------------------------------------------------------
        Returned variables:
        - err_test: vector containing the error done by the local regression 
          approximation (length equal to the regressand)
        - y_hat_te: (de-normalized) vector of the approximated regressand
        - w_hat_te: matrix whose columns are the weights vectors associated with each 
          row of the regressor matrix
        ------------------------------------------------------------------------------
        """
        # Find K nearest
        Np_test = self.Np_te      # n. patients in the test set
        Nf = self.Nf                            # number of features
        N = self.N

        # Isolate ndarrays of test set
        X_test_norm = np.copy(self.test_regressors_norm)
        y_test_norm = np.copy(self.test_regressand_norm)

        # Take training set (normalized values)
        X_tr = np.copy(self.train_regressors_norm)
        y_tr = np.copy(self.train_regressand_norm)

        # Iterate on all patients in the test set:
        for i in range(X_test_norm.shape[0]):
            x_current = X_test_norm[i, :]
            y_current = y_test_norm[i]

            distances = dist_eval(x_current, X_tr)

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
            self.w_hat_te[:, i] = np.copy(w_hat_te_curr)

            # Get y_hat of the current patient (x_curr) - normalized
            y_hat_te_curr_norm = np.reshape(
                x_current, (1, len(x_current))) @ np.reshape(w_hat_te_curr, (len(w_hat_te_curr), 1))

            # Store this result in element i of the vector self.y_hat
            # NOTE: after denormalizing
            self.y_hat_te[i] = np.copy(
                y_hat_te_curr_norm*self.stdev_regressand + self.mean_regressand)

            # Error - on de-normalized values
            self.err_test[i] = self.test_regressand[i] - self.y_hat_te[i]

        # Plot y vs. y_hat
        if plot_y:
            plt.figure(figsize=(6, 4))
            plt.plot(self.test_regressand, self.y_hat_te, '.')   # Place dots
            v = plt.axis()
            # Plot 45deg diagonal line
            plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
            plt.xlabel(r'$y$')
            plt.ylabel(r'$\^y$', rotation=0)
            plt.grid()
            plt.title(f"Local Linear Regression - Test set - N={self.N}")
            plt.tight_layout()
            if save_y:
                plt.savefig(imagepath_y)
            plt.show()

        # Plot error histogram
        if plot_hist:
            e = self.err_test

            plt.figure(figsize=(6, 4))
            plt.hist(e, bins=50, density=True, histtype='bar',
                     label='training')
            plt.xlabel(r"$e = y - \^y$")
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title(f'Local Linear Regression - Test error histogram - N={self.N}')
            plt.tight_layout()
            if save_hist:
                plt.savefig(imagepath_hist)
            plt.show()

        return self.err_test, self.y_hat_te, self.w_hat_te
    
    def errorAnalysis(self, plot_hist=False, save_hist=False, imagepath_hist='./img/15_LOCAL_error-hist_train-vs-test.png'):
        """
        errorAnalysis 
        ------------------------------------------------------------------------------
        Returns the error analysis for the Local Linear Regression model defined in 
        the LocalLR object, given the provided Test dataset.
        It is also possible to return the histogram comparing the error comparison 
        histogram, which displays the comparison between Training and Test datasets.
        ------------------------------------------------------------------------------
        Parameters:
        - test_regressand: DataFrame (Np x 1)
        - Test_regresors: DataFrame (Np x Nf)
        - plot_hist (default False): used to plot the histogram of the error
        - save_hist (default False)
        - imagepath_hist (default './img/LOCAL_error-hist_train-vs-test.png')
        ------------------------------------------------------------------------------
        Returned variable(s):
        - results_local: DataFrame containing the following error figures of merit, 
          for both the Training and Test datasets:
            - minimum value
            - maximum value
            - mean value
            - standard deviation
            - mean square error
            - coefficient of determination (wrt actual regressand values)
            - correlation coefficient (between actual regressand and approximated one)
        ------------------------------------------------------------------------------
        """
        
        if (self.w_hat_tr == 0).all():
            self.solve()
        
        if (self.w_hat_te == 0).all():
            self.test()

        err = [self.err_train, self.err_test]

        if plot_hist:
            plt.figure(figsize=(8, 4))
            plt.hist(err, bins=50, density=True, histtype='bar',
                     label=['Training set', 'Test set'])
            plt.xlabel(r"$e = y - \^y$")
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title(
                f'Local LR error histogram - comparison between training and test set - N={self.N}')
            plt.tight_layout()
            if save_hist:
                plt.savefig(imagepath_hist)
            plt.show()

        # Produce DF
        y_te = self.test_regressand

        # Analysis for SD
        # Training set
        E_tr_min = self.err_train.min()
        E_tr_max = self.err_train.max()
        E_tr_mu = self.err_train.mean()
        E_tr_sigma = self.err_train.std()
        E_tr_MSE = np.mean(self.err_train**2)
        # R^2 (coefficient of determination)
        R2_tr = 1 - E_tr_MSE/(np.std(self.train_regressand)**2)
        # Correlation coefficient
        c_tr = np.mean((self.train_regressand - self.mean_regressand)*(self.y_hat_tr - self.y_hat_tr.mean())
                       )/(self.stdev_regressand*self.y_hat_tr.std())

        # Test set
        E_te_min = self.err_test.min()
        E_te_max = self.err_test.max()
        E_te_mu = self.err_test.mean()
        E_te_sigma = self.err_test.std()
        E_te_MSE = np.mean(self.err_test**2)
        # R^2 (coefficient of determination)
        R2_te = 1 - E_te_MSE/(np.std(y_te)**2)
        # Correlation coefficient
        c_te = np.mean((y_te - y_te.mean())*(self.y_hat_te - self.y_hat_te.mean())
                       )/(y_te.std()*self.y_hat_te.std())

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
