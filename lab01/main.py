# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sub.minimization as mymin
import sub.linearRegression as myLR

#%%#### Preparing and analyzing the data: #################################

# Read Parkinson's data file and store result in
# a DataFrame (data structure defined by Pandas)
x = pd.read_csv("data/parkinsons_updrs.csv")

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
plt.title("Covariance of the features")
plt.tight_layout()
plt.savefig("./img/00_cov_matr.png")  # Save the figure
plt.show()

# Plot relationship between total UPDRS and the other features
plt.figure()
c.total_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)), features, rotation=90)
plt.title('Correlation coefficient between total UPDRS and other features')
plt.tight_layout()
plt.savefig("./img/00_1_corr_tot_UPDRS.png")
plt.show()

# Keep in mind: even though the patient ID seems correlated to the total
# UPDRS, it is not correct to use it as a regressor (it does not depend
# on the disease level in any way)

# It is convenient to shuffle the rows of the data - in this way we can
# prevent that only some patients are in the training set
# X --> Xsh
# Shuffling is random - set the seed
np.random.seed(315054)
# np.random.seed(1)

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
LR = myLR.LinearRegression(y_tr, X_tr, y_te, X_te)

# Solve Linear Regression using both LLS and Steepest Descent, then
# compare the resulting w_hat by plotting
LR.solve_LLS(plot_y=True, save_y=True)
LR.solve_SteepestDescent(stoppingCondition='iterations',
                         Nit=200, plot_y=True, save_y=True)
LR.plot_w(save_png=True)

# Performance evaluation - using test set
LR.LLS_test(plot_hist=True, save_hist=True, plot_y=True, save_y=True)
LR.SD_test(plot_hist=True, save_hist=True, plot_y=True, save_y=True)

error_vect = LR.test(plot_hist=True, save_hist=True)

finalResults = LR.errorAnalysis()
print("\nError analysis:")
print(finalResults)

#%%############# PART 2 - LOCAL LINEAR REGRESSION ##############################
N_closest = [10, 20, 50, 100]
# N_closest = [20]

size = X_tr.shape[0]
print(f"N. of patients in training set: {size}")

results_local = []

for N in N_closest:
    LocalLinearRegression = myLR.LocalLR(y_tr, X_tr, y_te, X_te, N)
    train_error_matrix = LocalLinearRegression.solve(
        plot_y=True, save_y=True, imagepath_y=f"./img/11_LOCAL_training-y_vs_y_hat_N{N}.png", plot_hist=True, save_hist=True, imagepath_hist=f'./img/12_LOCAL-training_err_hist_N{N}.png')[0]

# Evaluate performance on test set
    LocalLinearRegression.test(
        plot_y=True, save_y=True, imagepath_y=f"./img/13_LOCAL_test-y_vs_y_hat_N{N}.png", plot_hist=True, save_hist=True, imagepath_hist=f'./img/14_LOCAL-test_err_hist_N{N}.png')
    results_N = LocalLinearRegression.errorAnalysis(
        plot_hist=True, save_hist=True, imagepath_hist=f'./img/15_LOCAL_error-hist_train-vs-test_N{N}.png')
    results_local.append(results_N)

    print(f"\nN = {N}")
    print(results_N)

# Results local will contain all DataFrames, associated with each value of N_closest
#
#
#
#
#
#
#
#
#
#
#
#
#
#%%############# PART 3 - AVERAGING RESULTS OVER 20 SEEDS ##########################
#
# This section contains the same operations done so far on the data,
# repeated for different values of the seed

seeds = list(range(1, 21))

# Create containers for the variables that we need to average
# Results of the "traditional" linear regression (both LLS and Steepest Descent)
results_LR_20 = []
results_local_20 = []   # Results of the local linear regression

# Remove features in advance - NOTE: total_UPDRS was not removed since it will be removed later
X_str = X.drop(['subject#', 'Jitter:DDP', 'Shimmer:DDA'], axis=1)

for index in range(len(seeds)):

    # print(f"\nIteration number {index+1}: -----------------------------------------")
    # print(f"Seed = {seeds[index]}\n")

    np.random.seed(seeds[index])

    indexsh = np.arange(Np)         # Vector from 0 to Np-1
    np.random.shuffle(indexsh)      # Shuffle the vector of indices randomly
    Xsh = X_str.copy(deep=True)         # Copy X into Xsh

    # Shuffling (according to current seed)
    Xsh = Xsh.set_axis(indexsh, axis=0, copy=False)
    Xsh = Xsh.sort_index(axis=0)

    # 50% of shuffled matrix is out training set, other 50% is test set
    Ntr = int(0.5*Np)
    Nte = Np - Ntr

    # Isolate training data
    X_tr = Xsh[0:Ntr].drop(
        ['total_UPDRS'], axis=1)
    y_tr = Xsh['total_UPDRS'][0:Ntr]

    # Test set:
    X_te = Xsh[Ntr:].drop(
        ['total_UPDRS'], axis=1)
    y_te = Xsh['total_UPDRS'][Ntr:]

    # Create LinearRegression object
    LR = myLR.LinearRegression(y_tr, X_tr, y_te, X_te)

    # Solve Linear Regression using both LLS and Steepest Descent, then
    # compare the resulting w_hat by plotting
    LR.solve_LLS()
    LR.solve_SteepestDescent(stoppingCondition='iterations', Nit=200)

    # Performance evaluation - using test set
    LR.LLS_test()
    LR.SD_test()

    #error_vect = LR.test()

    finalResults = LR.errorAnalysis()
    # print("\nError analysis:")
    # print(finalResults)
    results_LR_20.append(finalResults)

    # Local linear regresison
    size = X_tr.shape[0]
    # print(f"N. of patients in training set: {size}")

    results_local = []

    for N in N_closest:
        LocalLinearRegression = myLR.LocalLR(y_tr, X_tr, y_te, X_te, N)
        train_error_matrix = LocalLinearRegression.solve()[0]

    # Evaluate performance on test set
        LocalLinearRegression.test()
        results_N = LocalLinearRegression.errorAnalysis()
        results_local.append(results_N)

        # print(f"N = {N}")
        # print(results_N)

    results_local_20.append(results_local)

# Averaging:

# "Traditional" LR:
# Extract matrix from each DF, sum them
# Divide by len(seeds)
tmp_mat_sum = np.zeros((results_LR_20[0].shape))
for i in range(len(seeds)):
    tmp_mat_sum += results_LR_20[i].values

combinations = [
    ('LLS', 'Training'), ('LLS', 'Test'),
    ('SD', 'Training'), ('SD', 'Test')
]
index_final = pd.MultiIndex.from_tuples(
    combinations, names=('Technique', 'Set'))

averaged_results_LR = pd.DataFrame(
    tmp_mat_sum/len(seeds), index=index_final, columns=finalResults.columns)

print("+-----------------------------------------------+")
print("| Averaged results - standard Linear Regression |")
print("+-----------------------------------------------+")
print(averaged_results_LR)
print('\n')

# Local LR:
# Extract corresponding elements for each value of N neighbors
print("+-----------------------------------------------+")
print("|  Averaged results - Local Linear regression   |")
print("+-----------------------------------------------+")

# Create empty list of tmp_sum matrices (one for each value of N used)
tmp_mat_N = [np.zeros((results_N.shape)) for ind in range(len(N_closest))]

for i in range(len(seeds)):
    for j in range(len(N_closest)):
        tmp_mat_N[j] += results_local_20[i][j].values

avg_mat_N = [mat/len(seeds) for mat in tmp_mat_N]

averaged_results_LocalLR = []
for i in range(len(N_closest)):
    tmp_df_loc = pd.DataFrame(
        avg_mat_N[i], index=results_N.index, columns=results_N.columns)
    averaged_results_LocalLR.append(tmp_df_loc)
    print(
        f"-------------------------------------------\nNumber of neighbors: {N_closest[i]}; Results:")
    print(tmp_df_loc)
