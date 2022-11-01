import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sub.minimization as mymin
import sub.linearRegression as myLR


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

finalResults = LR.errorAnalysis(y_te, X_te)
print(finalResults)
