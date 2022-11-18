import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sub.GPR as GPR
from sub.GPR import normalizeData


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
        raise ValueError(
            'Error! The number of features of the element is not the same as the one of the training set!')

    distance_vect = np.empty((train.shape[0],))

    for ind2 in range(train.shape[0]):
        tmp_sum = sum(np.power(element - train[ind2, :], 2))
        distance_vect[ind2] = np.sqrt(tmp_sum)

    return distance_vect


plt.close('all')

np.random.seed(315054)

inputFile = 'lab02/data/parkinsons_updrs.csv'
inputfile_2 = 'data/parkinsons_updrs.csv'

####### Preparing and analyzing the data: #################################

# Handle different file locations
try:
    x = pd.read_csv(inputFile)
except:
    x = pd.read_csv(inputfile_2)


# Print features name
features = list(x.columns)
print(features, "\n")

subj = pd.unique(x[features[0]])    # Distinct values of patient ID
n_subj = len(subj)

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

Np, Nc = X.shape    # Np: n. of patients, Nc: number of regressors Nf + 1

# Analyzing covariance could give strange results: test_time has a high variance
# It is necessary to first normalize the data, then evaluate the covariance (we
# are basically evaluating the correlation coefficient)

Xnorm = (X - X.mean())/X.std()  # Normalize the entire dataset
c = Xnorm.cov()  # Measure the covariance

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
Nval = int(0.25*Np)
Nte = Np - Ntr - Nval

# Isolate training data
X_tr = Xsh[['motor_UPDRS', 'age', 'PPE']][0:Ntr]
y_tr = Xsh['total_UPDRS'][0:Ntr]

# Validation set
X_val = Xsh[['motor_UPDRS', 'age', 'PPE']][Ntr:(Ntr+Nval)]
y_val = Xsh['total_UPDRS'][Ntr:(Ntr+Nval)]

# Test set:
X_te = Xsh[['motor_UPDRS', 'age', 'PPE']][(Ntr+Nval):]
y_te = Xsh['total_UPDRS'][(Ntr+Nval):]

Nf = X_tr.shape[1]
print(f"Training dataset dimensions: {Ntr} x {Nf}")
print(f"Validation dataset dimensions: {Nval} x {Nf}")
print(f"Test dataset dimensions: {Nte} x {Nf}")

# Normalization:
mean_X = np.mean(X_tr)
stdev_X = np.std(X_tr)
X_tr_norm = normalizeData(X_tr, mean_X, stdev_X).values

mean_y = np.mean(y_tr)
stdev_y = np.std(y_tr)
y_tr_norm = normalizeData(y_tr, mean_y, stdev_y).values

X_val_norm = normalizeData(X_val, mean_X, stdev_X).values
y_val_norm = normalizeData(y_val, mean_y, stdev_y).values

X_te_norm = normalizeData(X_te, mean_X, stdev_X).values
y_te_norm = normalizeData(y_te, mean_y, stdev_y).values

######## GPR ######################
r2 = 1
theta = 1
s2 = 0.001

y_hat_val_norm = np.zeros((len(y_val_norm, )))

N = 10

for k in range(len(y_val)):
    x = X_val_norm[k, :]
    y = y_val_norm[k]

    # Find N-1 closest elements in the training set
    dist_vec = dist_eval(x, X_tr_norm)

    closest_ind = np.argsort(dist_vec)[:10]

    X_Nmin1 = X_tr_norm[closest_ind, :]
    y_Nmin1 = y_tr_norm[closest_ind]

    GaussReg = GPR.GaussianProcessRegression(
        y_Nmin1, X_Nmin1, y, x, r2, theta, s2, normalize=False)

    y_hat = GaussReg.solve()

    y_hat_val_norm[k] = y_hat

plt.figure(figsize=(12, 6))
plt.plot(y_val_norm, y_hat_val_norm, '.')
v = plt.axis()
# Plot 45deg diagonal line
plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
plt.grid()
plt.xlabel(r"$y$")
plt.ylabel(r"$\hat{y}$")
plt.title(
    f"Validation set - r^2 = {r2}, theta = {theta}, sigma_nu = {s2}")
plt.show()
