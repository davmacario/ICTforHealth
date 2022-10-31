import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sub.minimization as mymin
import sub.linearRegression as myLR


# TODO [1]: Introduce classes and methods for plotting graphs and generating
# resulting dataframes

# TODO [2]: Compare Steepest Descent and LLS results
# TODO [3]: local linear regression model (find 10 neighbors)
# TODO [4]: plot:
#          - Estimated regressand vs. true regressand (de-norm)
#          - Histogram of de-norm. estimation error
# TODO [5]: fill DataFrame with the measured min, max, mean, stdev, msv, R^2,
# correlation coeff. for regression errors, comparing normal and local
# linear regression
# TODO [6]: run the program also with 20 values of the seed and average results


#### Preparing and analyzing the data: #################################

# Read Parkinson's data file and store result in
# a DataFrame (data structure defined by Pandas)
x = pd.read_csv("lab01/data/parkinsons_updrs.csv")

# Check data
x.describe().T
x.info()

# Check names/meanings of the available features
features = list(x.columns)
print(features, "\n")

# For each patient there are multiple observations
# We can look at how many patients are present:
subj = pd.unique(x[features[0]])    # Distinct values of patient ID
n_subj = len(subj)
print(f"The number of distinct patients in the dataset is {n_subj}")

# Evaluate the daily average of the features for each patient
X = pd.DataFrame()

for k in subj:
    xk = x[x[features[0]] == k]    # xk contains all measurements of patient k
    xk1 = xk.copy()

    # Remove decimal value - method astype is used to cast DataFrames to a specific type
    # Integer value is the day ID
    xk1.test_time = xk1.test_time.astype(int)
    xk1['g'] = xk1['test_time']  # Add new feature 'g'
    v = xk1.groupby('g').mean()  # Group measurements related to the same day
    # feature 'g' is then removed

    # Append the new data to X
    X = pd.concat([X, v], axis=0, ignore_index=True)

print("The dataset shape after the mean is: ", X.shape)
print("The features of the dataset are: ", len(features))
print(features)

Np, Nc = X.shape    # Np: n. of patients, Nc: number of regressors Nf + 1

# Analyzing covariance could give strange results: test_time has a high variance
# It is necessary to first normalize the data, then evaluate the covariance (we
# are basically evaluating the correlation coefficient)

Xnorm = (X - X.mean())/X.std()  # Normalize the entire dataset
c = Xnorm.cov()  # Measure the covariance

# Plot result
plt.figure()
plt.matshow(np.abs(c.values), fignum=0)
plt.xticks(np.arange(len(features)), features, rotation=90)
plt.yticks(np.arange(len(features)), features, rotation=0)
plt.colorbar()
plt.title("Correlation coefficients of the features")
plt.tight_layout()
plt.savefig("./lab01/img/corr_coeff.png")  # Save the figure
plt.show()

# Plot relationship between total UPDRS and the other features
plt.figure()
c.total_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)), features, rotation=90)
plt.title('Correlation coefficient between total UPDRS and other features')
plt.tight_layout()
plt.savefig("./lab01/img/corr_tot_UPDRS.png")
plt.show()

# Keep in mind: even though the patient ID seems correlated to the total
# UPDRS, it is not correct to use it as a regressor (it does not depend
# on the disease level in any way)

# It is convenient to shuffle the rows of the data - in this way we can
# prevent that only some patients are in the training set
# X --> Xsh
# Shuffling is random - set the seed
np.random.seed(315054)
indexsh = np.arange(Np)  # Vector from 0 to Np-1
np.random.shuffle(indexsh)      # Shuffle the vector of indices randomly
Xsh = X.copy(deep=True)       # Copy X into Xsh

# Shuffling of Xsh is performed by assigning the indices of the shuffled
# vector to the rows of Xsh and then performing a sort (!!!)
Xsh = Xsh.set_axis(indexsh, axis=0, inplace=False)
Xsh = Xsh.sort_index(axis=0)

########################################################################
# REGRESSAND: Total UPDRS
# REGRESSORS: all other features, excluding patient ID, Jitter:DDP and
# Shimmer:DDA
########################################################################

#%% ### Performing Regression #############################################

########################################################################################
# From this point on, the script needs to be rewritten using the class

# 50% of shuffled matrix is out training set, other 50% is test set
Ntr = int(0.5*Np)
Nte = Np - Ntr

# Isolate training data
X_tr = Xsh[0:Ntr].drop(
    ['total_UPDRS', 'subject#', 'Jitter:DDP', 'Shimmer:DDA'], axis=1)
y_tr = Xsh['total_UPDRS'][0:Ntr]

# Test set:
X_te = Xsh[Ntr:].drop(
    ['total_UPDRS', 'subject#', 'Jitter:DDP', 'Shimmer:DDA'], axis=1)
y_te = Xsh['total_UPDRS'][Ntr:]

# Create LinearRegression object
LR = myLR.LinearRegression(y_tr, X_tr)

# Solve Linear Regression using both LLS and Steepest Descent, then
# compare the resulting w_hat by plotting
LR.solve_LLS(plot_y=True, save_y=True)
LR.solve_SteepestDescent(Nit=50, plot_y=True, save_y=True)
LR.plot_w(save_png=True)

# Performance evaluation - using test set
LR.LLS_test(y_te, X_te, plot_hist=True, plot_y=True)
LR.SD_test(y_te, X_te, plot_hist=True, plot_y=True)

error_vect = LR.test(y_te, X_te, plot_hist=True, save_hist=True)

# %%

# Performance evaluation
# Now, evaluate y_hat
y_hat_tr_norm = X_tr_norm@w_hat
y_hat_te_norm = X_te_norm@w_hat

# Now we can de-normalize (normalized values don't have a medical meaning)
y_hat_tr = y_hat_tr_norm*sy + my
y_tr = y_tr_norm*sy + my

y_hat_te = y_hat_te_norm*sy + my
y_te = y_te_norm*sy + my

# Check trands in the regression error (y - y_hat)
# To do so, use a histogram
E_tr = (y_tr - y_hat_tr)
E_te = (y_te - y_hat_te)

e = [E_tr, E_te]        # It is a list containing 2 ndarrays (not merged)

plt.figure(figsize=(6, 4))
plt.hist(e, bins=50, density=True, histtype='bar', label=['training', 'test'])
plt.xlabel(r"$e = y - \^y$")
plt.ylabel(r'$P(e$ in bin$)$')
plt.legend()
plt.grid()
plt.title('LLS - Error histogram')
plt.tight_layout()
plt.savefig('./lab01/img/LLS-err_hist.png')
plt.show()

# Now it can be useful to evlauate the performance by plotting the actual
# values of y_te vs. the estimated ones (my means of lin. reg.)

plt.figure(figsize=(6, 4))
plt.plot(y_te, y_hat_te, '.')   # Place dots
v = plt.axis()
# Plot 45deg diagonal line
plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
plt.xlabel(r'$y$')
plt.ylabel(r'$\^y$')
plt.grid()
plt.title("LLS - Test")
plt.tight_layout()
plt.savefig('./lab01/img/LLS-yhat-vs-y.png')
plt.show()

# Evaluate other parameters - max, min, stdev, msv of the error, ...
# Over training set
E_tr_min = E_tr.min()
E_tr_max = E_tr.max()
E_tr_mu = E_tr.mean()
E_tr_sigma = E_tr.std()
E_tr_MSE = np.mean(E_tr**2)
# R^2 (coefficient of determination)
R2_tr = 1 - E_tr_MSE/(np.std(y_tr**2))
# Correlation coefficient
c_tr = np.mean((y_tr - y_tr.mean())*(y_hat_tr - y_hat_tr.mean())
               )/(y_tr.std()*y_hat_tr.std())

# Over test set
E_te_min = E_te.min()
E_te_max = E_te.max()
E_te_mu = E_te.mean()
E_te_sigma = E_te.std()
E_te_MSE = np.mean(E_te**2)
# R^2 (coefficient of determination)
R2_te = 1 - E_te_MSE/(np.std(y_te**2))
# Correlation coefficient
c_te = np.mean((y_te - y_te.mean())*(y_hat_te - y_hat_te.mean())
               )/(y_te.std()*y_hat_te.std())

# Put together these performance figures, create a DataFrame
rows = ['Training', 'Test']
cols = ['min', 'max', 'mean', 'std', 'MSE', 'R^2', 'corr_coeff']
p = np.array([
    [E_tr_min, E_tr_max, E_tr_mu, E_tr_sigma, E_tr_MSE, R2_tr, c_tr],
    [E_te_min, E_te_max, E_te_mu, E_te_sigma, E_te_MSE, R2_te, c_te]
])
results = pd.DataFrame(p, columns=cols, index=rows)

print(results)
