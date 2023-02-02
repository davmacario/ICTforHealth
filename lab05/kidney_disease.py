from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import sklearn.ensemble as ske
import seaborn as sn

pd.set_option('display.precision', 3, 'display.max_columns',
              50, 'display.max_rows', 50)
np.set_printoptions(precision=3)
# np.set_printoptions(linewidth=100)

# %%
# define the feature names:
feat_names = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc',
              'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
              'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe',
              'ane', 'classk']
ff = np.array(feat_names)
feat_cat = np.array(['num', 'num', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat',
                     'num', 'num', 'num', 'num', 'num', 'num', 'num', 'num', 'num',
                     'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat'])
# import the dataframe:
# xx = pd.read_csv("./data/chronic_kidney_disease.arff", sep=',',
#                  skiprows=29, names=feat_names,
#                  header=None, na_values=['?', '\t?'],
#                  warn_bad_lines=True)
xx = pd.read_csv("./data/chronic_kidney_disease_v2.arff", sep=',',
                 skiprows=29, names=feat_names,
                 header=None, na_values=['?', '\t?'])

Np, Nf = xx.shape

# %% change categorical data into numbers:
mapping = {
    'normal': 0,
    'abnormal': 1,
    'present': 1,
    'notpresent': 0,
    'yes': 1,
    ' yes': 1,
    'no': 0,
    '\tno': 0,
    '\tyes': 1,
    'ckd': 1,
    'notckd': 0,
    'poor': 1,
    'good': 0,
    'ckd\t': 1}

xx = xx.replace(mapping.keys(), mapping.values())

# Alternatively:
# key_list=["normal","abnormal","present","notpresent","yes",
# "no","poor","good","ckd","notckd","ckd\t","\tno"," yes","\tyes"]
# key_val=[0,1,0,1,0,1,0,1,1,0,1,1,0,0]
# xx=xx.replace(key_list,key_val)
# show the cardinality of each feature in the dataset; in particular classk should have only two possible values
print(f"Cardinality of the features{xx.nunique()}")
print(f"\nInformation about the data set: {xx.info()}")

# %%  Manage the missing data through regression ################################
x = xx.copy()

# Drop rows with less (<=) than 19 = Nf-6 recorded features:
min_n_values = Nf - 6
x = x.dropna(thresh=min_n_values)   # Thresh is the min. number of non-missing
# necessary to have index without "jumps"
x.reset_index(drop=True, inplace=True)

n = x.isnull().sum(axis=1)  # check the number of missing values in each row
print('Max number of missing values in the reduced dataset: ', n.max())
print('Number of points in the reduced dataset: ', len(n))

# Take the rows with exctly Nf=25 useful features;
# this is going to be the training dataset for regression
Xtrain = x.dropna(thresh=Nf)
Xtrain.reset_index(drop=True, inplace=True)  # reset the index of the dataframe

# Get the possible values (i.e. alphabet) for the categorical features
alphabets = []
for k in range(len(feat_cat)):
    if feat_cat[k] == 'cat':
        val = Xtrain.iloc[:, k]
        val = val.unique()
        alphabets.append(val)
    else:
        alphabets.append('num')

# Run regression tree on all the missing data
# Normalize the training dataset
mm = Xtrain.mean(axis=0)
ss = Xtrain.std(axis=0)
Xtrain_norm = (Xtrain-mm)/ss

# %% REGRESSION ################################################################
# Get test dataset by removing rows with no null elements
Xtest = x.drop(x[x.isnull().sum(axis=1) == 0].index)
Xtest.reset_index(drop=True, inplace=True)  # reset the index of the dataframe

# Normalize the test dataset
Xtest_norm = (Xtest-mm)/ss
Np, Nf = Xtest_norm.shape

regr = tree.DecisionTreeRegressor()     # instantiate the regressor
for kk in range(Np):
    xrow = Xtest_norm.iloc[kk]          # k-th row
    mask = xrow.isna()                  # columns with nan in row kk

    # remove the columns from the training dataset

    # For each element, we take the training dataset columns we need
    # and perform regression to find the values that are missing
    Data_tr_norm = Xtrain_norm.loc[:, ~mask]
    y_tr_norm = Xtrain_norm.loc[:, mask]       # columns to be regressed

    # Apply regression via
    regr_fit = regr.fit(Data_tr_norm, y_tr_norm)   # find the regression tree
    Data_te_norm = Xtest_norm.loc[kk, ~mask].values.reshape(1, -1)
    ytest_norm = regr_fit.predict(Data_te_norm)

    # Rewrite the row, by adding the values obtained with regression
    a = xrow.values.astype(float)               # Cast
    a[mask] = ytest_norm.flatten()              # Insert regressed values
    # substitute nan with regressed values
    Xtest_norm.iloc[kk] = a

Xtest_new = Xtest_norm*ss+mm                    # denormalize

# %% Substitute regressed numerical values with the closest element in the alphabet
# Determine where are the categorical features
index = np.argwhere(feat_cat == 'cat').flatten()
for k in index:
    val = alphabets[k]              # possible values for the feature
    c = Xtest_new.iloc[:, k].values  # values in the column
    c = c.reshape(-1, 1)            # column vector
    val = val.reshape(1, -1)        # row vector
    # matrix with all the distances w.r.t. the alphabet values
    d = (val-c)**2
    # find the index of the closest alphabet value
    ii = d.argmin(axis=1)
    Xtest_new.iloc[:, k] = val.flatten()[ii]

# %% get the new dataset with no missing values
X_new = pd.concat([Xtrain, Xtest_new], ignore_index=True, sort=False)
X_new.to_csv('./data/kidney_no_nan.csv')

# %% check the cumulative distribution functions
L = X_new.shape[0]

plotCDF = True

if plotCDF:
    for k in range(Nf):
        plt.figure()
        a = xx.iloc[:, k].dropna()
        M = a.shape[0]
        plt.plot(np.sort(a), np.arange(M)/M, label='original dataset')
        plt.plot(np.sort(X_new.iloc[:, k]), np.arange(
            L)/L, label='regressed dataset')
        plt.title('CDF of '+xx.columns[k])
        plt.xlabel('x')
        plt.ylabel('P(X<=x)')
        plt.grid()
        plt.legend(loc='upper left')
    plt.show()

# %%------------------ Decision tree -------------------

decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
random_forest = ske.RandomForestClassifier(
    n_estimators=150, criterion='entropy')

np.random.seed = 315054

N_iter = 3

acc_tree = []
conf_tree = []

acc_forest = []
conf_forest = []

for i in range(N_iter):
    # Perform N iterations choosing a random seed
    r = np.random.randint(100)
    print('+------------------+')
    print(f'| Random Seed: {r:3} |')
    print('+------------------+')

    Xsh = X_new.sample(frac=1, replace=False, random_state=r,
                       axis=0, ignore_index=True)

    Ntrain = L//2

    XshTrain = Xsh[Xsh.index < Ntrain]
    XshTest = Xsh[Xsh.index >= Ntrain]

    y_tr = XshTrain['classk']
    X_tr = XshTrain.drop('classk', axis=1)

    decision_tree_fitted = decision_tree.fit(X_tr, y_tr)
    y_hat_te = decision_tree_fitted.predict(XshTest.drop('classk', axis=1))

    acc_tree.append(accuracy_score(XshTest['classk'], y_hat_te))
    conf_tree.append(confusion_matrix(XshTest['classk'], y_hat_te))

    print('Random Tree:')
    print('Accuracy = ', acc_tree[i])
    print(conf_tree[i])

    target_names = ['notckd', 'ckd']

    fig, axes = plt.subplots()
    tree.plot_tree(decision_tree_fitted,
                   feature_names=feat_names[:24],
                   class_names=target_names,
                   proportion=False,
                   filled=True)

    plt.title('shuffled_data (seed ' + str(r) + ')')
    plt.savefig('./img/tree_shuffle_' + str(r) + '.png')
    plt.show()

    ##############################################################
    # Random Forest Classifier

    random_forest_fitted = random_forest.fit(X_tr, y_tr)

    y_hat_te_for = random_forest_fitted.predict(XshTest.drop('classk', axis=1))

    acc_forest.append(accuracy_score(XshTest['classk'], y_hat_te_for))
    conf_forest.append(confusion_matrix(XshTest['classk'], y_hat_te_for))

    print('Random Forest:')
    print('Accuracy = ', acc_forest[i])
    print(conf_forest[i])
