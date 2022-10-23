import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sub.minimization as mymin

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

    xk1.test_time = xk1.test_time.astype(int)   # Remove decimal value
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
plt.title('Correlation coefficient among total UPDRS and other features')
plt.tight_layout()
plt.savefig("./lab01/img/corr_tot_UPDRS.png")
plt.show()

# Keep in mind: even though the patient ID seems correlated to the total
# UPDRS, itis not correct to use it as a regressor (it does not depend
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
# REGRESSORS: all other features, excluding patient ID
########################################################################

#### Performing Regression #############################################

# First, perform normalization in order to obtain zero mean and unit
# variance in the features (which speeds up computation)
# --> Subtract the mean and divide by std. dev. each feature

# NOTE: features mean and standard dev. can only be measured from the
# training dataset!

# 50% of shuffled matrix is out training set, other 50% is test set
Ntr = int(0.5*Np)
Nte = Np - Ntr

# Isolate training data
X_tr = Xsh[0:Ntr]

mm = X_tr.mean()        # Mean of the training data (for all features)
ss = X_tr.std()         # Std. dev. for all features

my = mm.total_UPDRS     # Mean of total UPDRS
sy = ss.total_UPDRS     # Std. dev. of total UPDRS

# Normalize data
Xsh_norm = (Xsh-mm)/ss  # Normalize data
ysh_norm = Xsh_norm['total_UPDRS']  # Extract regressand
# Isolate regressors (remove total UPDRS and patient ID)
# Also remove jitterDDP and Shimmer DDA
Xsh_norm = Xsh_norm.drop(
    ['total_UPDRS', 'subject#', 'Jitter:DDP', 'Shimmer:DDA'], axis=1)

# Obtain final training and test datasets
X_tr_norm = Xsh_norm[0:Ntr]
X_te_norm = Xsh_norm[Ntr:]

y_tr_norm = ysh_norm[0:Ntr]
y_te_norm = ysh_norm[Ntr:]

# In order to solve the Linear Least Squares problem to find the
# vector of weights w_hat, it is easier to work with Ndarrays
# (NumPy)

X_tr_norm = X_tr_norm.values
X_te_norm = X_te_norm.values

y_tr_norm = y_tr_norm.values
y_te_norm = y_te_norm.values

# %%
# Solve with LLS
# w_hat = np.linalg.inv(X_tr_norm.T@X_tr_norm)@(X_tr_norm.T@y_tr_norm)


# %%
# Solve regression with Steepest Descent
w_SD = mymin.SteepestDescent(y_tr_norm, X_tr_norm)
w_hat = w_SD.run(Nit=50)

# Plot w_hat (look at potential problems, e.g., collinearity)
regressors = list(Xsh_norm.columns)     # Get list of regressors
# Number of features = length of w_hat = len(regressors)
Nf = len(w_hat)

nn = np.arange(Nf)

plt.figure(figsize=(6, 4))
plt.plot(nn, w_hat, '-o')
ticks = nn
# Each tick will have the label of the corresp. regressor
plt.xticks(ticks, regressors, rotation=90)
plt.ylabel(r'$\^w(n)$')     # Using LaTex notation
plt.title('LLS - Optimized weights')
plt.grid()
plt.tight_layout()
plt.savefig("./lab01/img/LLS-w_hat.png")
plt.show()

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

e = [E_tr, E_te]        # It is a list containing 2 lists (not merged)

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
