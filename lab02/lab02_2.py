import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sub.GPR as GPR
from sub.GPR import normalizeData, denormalizeData


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

inputFile = './lab02/data/parkinsons_updrs.csv'
inputfile_2 = './data/parkinsons_updrs.csv'

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

Xnorm = (X - X.mean(axis=0))/X.std(axis=0)  # Normalize the entire dataset
c = Xnorm.cov()  # Measure the covariance

indexsh = np.arange(Np)  # Vector from 0 to Np-1
np.random.shuffle(indexsh)      # Shuffle the vector of indices randomly
Xsh = X.copy(deep=True)       # Copy X into Xsh

# Shuffling of Xsh is performed by assigning the indices of the shuffled
# vector to the rows of Xsh and then performing a sort (!!!)
Xsh = Xsh.set_axis(indexsh, axis=0, copy=False)
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
X_tr = Xsh[['motor_UPDRS', 'age', 'PPE']].values[0:Ntr]
y_tr = Xsh['total_UPDRS'].values[0:Ntr]

# Validation set
X_val = Xsh[['motor_UPDRS', 'age', 'PPE']].values[Ntr:(Ntr+Nval)]
y_val = Xsh['total_UPDRS'].values[Ntr:(Ntr+Nval)]

# Test set:
X_te = Xsh[['motor_UPDRS', 'age', 'PPE']].values[(Ntr+Nval):]
y_te = Xsh['total_UPDRS'].values[(Ntr+Nval):]

Nf = X_tr.shape[1]
print(f"Training dataset dimensions: {Ntr} x {Nf}")
print(f"Validation dataset dimensions: {Nval} x {Nf}")
print(f"Test dataset dimensions: {Nte} x {Nf}")

# Normalization:
mean_X = np.mean(X_tr)
stdev_X = np.std(X_tr)
X_tr_norm = normalizeData(X_tr, mean_X, stdev_X)

mean_y = np.mean(y_tr)
stdev_y = np.std(y_tr)
y_tr_norm = normalizeData(y_tr, mean_y, stdev_y)

X_val_norm = normalizeData(X_val, mean_X, stdev_X)
y_val_norm = normalizeData(y_val, mean_y, stdev_y)

X_te_norm = normalizeData(X_te, mean_X, stdev_X)
y_te_norm = normalizeData(y_te, mean_y, stdev_y)

######## GPR ######################
theta = 1

# Grid search for the best hyperparameters (not theta
# since the data will be normalized and 1 is okay)
r2 = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.80, 0.9, 1, 1.1, 1.2])
s2 = np.array([0.0001, 0.00025, 0.0005, 0.0006, 0.0007, 0.0008])

# Parameter grid
param_grid = np.zeros((len(r2), len(s2), 2))

for i in range(len(r2)):
    for j in range(len(s2)):
        param_grid[i, j, 0] = r2[i]
        param_grid[i, j, 1] = s2[j]

flat_grid = param_grid.reshape((len(r2)*len(s2), 2))

N = 10

# When iterating on the parameter grid, need to store
# each resulting y_hat_val
y_hat_val_list = []
mse_val_list = []

for ind in range(flat_grid.shape[0]):
    r2_curr = flat_grid[ind][0]
    s2_curr = flat_grid[ind][1]

    y_hat_val_norm = np.zeros((len(y_val_norm, )))

    # Initial procedure to check it all works
    for k in range(len(y_val)):
        x = X_val_norm[k, :]
        y = y_val_norm[k]

        # Find N-1 closest elements in the training set
        dist_vec = dist_eval(x, X_tr_norm)

        closest_ind = np.argsort(dist_vec)[:10]

        X_Nmin1 = X_tr_norm[closest_ind, :]
        y_Nmin1 = y_tr_norm[closest_ind]

        GaussReg = GPR.GaussianProcessRegression(
            y_Nmin1, X_Nmin1, y, x, r2_curr, theta, s2_curr, normalize=False)

        y_hat = GaussReg.solve()[0]

        y_hat_val_norm[k] = y_hat

    # De-normalize data
    y_hat_val = denormalizeData(y_hat_val_norm, mean_y, stdev_y)
    y_hat_val_list.append(y_hat_val)

    # Evaluate the mean square error -> metric to be minimized by
    # choosing the right hyperparameters (grid search)
    mse_val = sum((y_val - y_hat_val)**2)/len(y_hat_val)
    mse_val_list.append(mse_val)

#     plt.figure(figsize=(12, 6))
#     plt.plot(y_val_norm, y_hat_val_norm, '.')
#     v = plt.axis()
#     # Plot 45deg diagonal line
#     plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
#     plt.grid()
#     plt.xlabel(r"$y$")
#     plt.ylabel(r"$\hat{y}$")
#     plt.title(
#         f"Validation set - r^2 = {r2}, theta = {theta}, sigma_nu = {s2}")

# plt.show()

# Best parameters among the considered ones:
ind_best = np.argmin(mse_val_list)
best_params = flat_grid[ind_best, :]

print(f"Best parameters: {best_params}\nLowest MSE: {mse_val_list[ind_best]}")

r2_best = best_params[0]
s2_best = best_params[1]


print(f"Used parameters:\n\
N = {N}         \n\
theta = {theta} \n\
r^2 = {r2_best} \n\
s^2 = {s2_best} \n"
      )

########## Training set ##########

y_hat_tr_norm = np.zeros((len(y_tr_norm, )))

for k in range(len(y_tr)):
    x = X_tr_norm[k, :]
    y = y_tr_norm[k]

    # Find N-1 closest elements in the training set
    dist_vec_tr = dist_eval(x, X_tr_norm)

    closest_ind_tr = np.argsort(dist_vec_tr)[:10]

    X_Nmin1 = X_tr_norm[closest_ind_tr, :]
    y_Nmin1 = y_tr_norm[closest_ind_tr]

    GaussReg = GPR.GaussianProcessRegression(
        y_Nmin1, X_Nmin1, y, x, r2_curr, theta, s2_curr, normalize=False)

    y_hat = GaussReg.solve()[0]

    y_hat_tr_norm[k] = y_hat

y_hat_tr = denormalizeData(y_hat_tr_norm, mean_y, stdev_y)

# Performance evaluation - testing phase
e_tr = y_tr - y_hat_tr      # Estimation error - test set
mean_e_tr = e_tr.mean(axis=0)
stdev_e_tr = e_tr.std(axis=0)
msv_e_tr = (e_tr**2).mean(axis=0)
e_tr_R2 = 1 - msv_e_tr/(np.std(y_tr)**2)

########## Validation set ##########
# Error - validation set with optimal parameters
e_val = y_val - y_hat_val_list[ind_best]
mean_e_val = e_val.mean(axis=0)
stdev_e_val = e_val.std(axis=0)
msv_e_val = (e_val**2).mean(axis=0)
e_val_R2 = 1 - msv_e_val/(np.std(y_val)**2)

########## Test set ##########

y_hat_te_norm = np.zeros((len(y_te_norm, )))
stdev_y_hat_te_norm = np.zeros((len(y_te_norm, )))

for k in range(len(y_te)):
    x = X_te_norm[k, :]
    y = y_te_norm[k]

    # Find N-1 closest elements in the training set
    dist_vec_te = dist_eval(x, X_tr_norm)

    closest_ind_te = np.argsort(dist_vec_te)[:10]

    X_Nmin1 = X_tr_norm[closest_ind_te, :]
    y_Nmin1 = y_tr_norm[closest_ind_te]

    GaussReg = GPR.GaussianProcessRegression(
        y_Nmin1, X_Nmin1, y, x, r2_curr, theta, s2_curr, normalize=False)

    y_hat, stdev_y_hat = GaussReg.solve()

    y_hat_te_norm[k] = y_hat
    stdev_y_hat_te_norm[k] = stdev_y_hat

y_hat_te = denormalizeData(y_hat_te_norm, mean_y, stdev_y)
# Need also to denormalize the stdev, i.e., multiply it by the stdev of y
stdev_y_hat_te = denormalizeData(stdev_y_hat_te_norm, 0, stdev_y)

# Performance evaluation - testing phase
e_te = y_te - y_hat_te      # Estimation error - test set
mean_e_te = e_te.mean(axis=0)
stdev_e_te = e_te.std(axis=0)
msv_e_te = (e_te**2).mean(axis=0)
e_te_R2 = 1 - msv_e_te/(np.std(y_te)**2)

# Print results
cols = ['mean', 'stdev', 'MSE', 'R2']
index = ['train', 'val', 'test']

gpr_res_DF = pd.DataFrame(
    [
        [mean_e_tr, stdev_e_tr, msv_e_tr, e_tr_R2],
        [mean_e_val, stdev_e_val, msv_e_val, e_val_R2],
        [mean_e_te, stdev_e_te, msv_e_te, e_te_R2]
    ], index=index, columns=cols
)

print('GPR results:')
print(gpr_res_DF)

# Error histogram
e_all = [e_tr, e_val, e_te]

plt.figure(figsize=(10, 6))
plt.hist(e_all, bins=50, density=True, histtype='bar',
         label=['train', 'validation', 'test'])
plt.xlabel(r"$e = y_{te} - \^y_{te}$")
plt.ylabel(r'$P(e$ in bin$)$')
plt.legend()
plt.grid()
plt.title(f'DPR - Error histogram - N={N}')
plt.tight_layout()
plt.savefig('./img/error_hist.png')
plt.show()

# Estimated UPDRS vs. true one, with error bars:
three_sigma = 3*stdev_y_hat_te

plt.figure(figsize=(10, 6))
plt.plot(y_te, y_hat_te, '.')
v = plt.axis()
# Plot 45deg diagonal line
plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
for i in range(len(y_te)):
    plt.plot(y_te[i]*np.ones((2,)), np.array(
        [y_hat_te[i] - three_sigma[i], y_hat_te[i] + three_sigma[i]]), 'g')
plt.xlabel(r'$y$')
plt.ylabel(r'$\^y$', rotation=0)
plt.grid()
plt.title(f"GPR - Test set - N={N}")
plt.tight_layout()
plt.savefig('./img/y_hat-vs-y_te.png')
plt.show()


# LLS
w_hat_LLS = np.linalg.inv(
    X_tr_norm.T@X_tr_norm)@(X_tr_norm.T@y_tr_norm)

y_hat_te_LLS_norm = X_te_norm@(w_hat_LLS)
y_hat_te_LLS = denormalizeData(y_hat_te_LLS_norm, mean_y, stdev_y)

error_te_LLS = y_te - y_hat_te_LLS
mean_e_te_LLS = error_te_LLS.mean(axis=0)
stdev_e_te_LLS = error_te_LLS.std(axis=0)
msv_e_te_LLS = (error_te_LLS**2).mean(axis=0)
e_te_R2_LLS = 1 - msv_e_te_LLS/(np.std(y_te)**2)

res_LLS = pd.DataFrame([[mean_e_te_LLS, stdev_e_te_LLS,
                       msv_e_te_LLS, e_te_R2_LLS]], index=['test'], columns=cols)
print('LLS results: ')
print(res_LLS)
