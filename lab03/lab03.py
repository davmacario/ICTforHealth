import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk


def findSS(y, x):
    # Sort values of x and use them as thresholds
    ind_sort = np.argsort(x)
    thresh = np.append([0], x[ind_sort])    # Notice: added 0

    sens_list = []
    spec_list = []

    x0 = x[y == 0]
    x1 = x[y == 1]

    Np = np.sum(swab==1)    # number of ill
    Nn = np.sum(swab==0)    # number of healthy

    for i in range(len(thresh)):
        thresh_now = thresh[i]
        
        n1 = np.sum(x1 > thresh_now)
        sens_list.append(n1/Np)

        n0 = np.sum(x0 < thresh_now)
        spec_list.append(n0/Nn)

    sensitivity = np.array(sens_list)
    specificity = np.array(spec_list)

    return thresh, sensitivity, specificity




plt.rcParams["font.family"] = "Times New Roman"
#%%
plt.close('all')

inputFile = 'data/covid_serological_results.csv'
inputFile_2 = 'lab03/data/covid_serological_results.csv'

try:
    xx = pd.read_csv(inputFile)
except:
    xx = pd.read_csv(inputFile_2)

features = xx.columns
print(features)

# 0 = no illness; 1 = unclear; 2 = ill
swab=xx.COVID_swab_res.values
swab[swab>=1] = 1# non reliable values are considered positive

Test1 = xx.IgG_Test1_titre.values
Test2 = xx.IgG_Test2_titre.values

## Plot test 2 vs. test 1 - see outliers
plt.figure()
plt.plot(Test1, Test2, '.')
plt.title("Visualize outliers")
plt.xlabel("Test 1")
plt.ylabel("Test 2")
plt.grid()
plt.show()

data = xx.values
N_points = data.shape[0]

data_norm = (data - data.mean())/data.std()       # Normalize data

## Carry out DBSCAN, setting suitable parameters (eps, M)
clustering = sk.DBSCAN(eps = 0.5, min_samples=4).fit(data_norm)

ii = np.argwhere(clustering.labels_ == -1)[:, 0]    # Outliers
# Print outlier
print(xx.iloc[ii])

## Drop the outliers
xx = xx.drop(ii)

# Re-extract data:
swab = xx.COVID_swab_res.values         # 0 = no illness; 1 = illness
Test1 = xx.IgG_Test1_titre.values
Test2 = xx.IgG_Test2_titre.values

############# Sensitivity and Specificity #######################################

############# Test1

############# Test2
x = Test2       
y = swab        # Correct outcomes

x0 = x[swab==0]     # Test results for healthy
x1 = x[swab==1]     # Test results for ill

Np = np.sum(swab==1)    # number of ill
Nn = np.sum(swab==0)    # number of healthy

# Set the thresh
thresh = 5

n1 = np.sum(x1 > thresh)
sens = n1/Np

n0 = np.sum(x0 < thresh)
spec = n0/Nn

print(f"Sensitivity - test 2: {sens}\nSpecificity - test 2: {spec}")

plt.figure()
x_hist = [x0, x1]
plt.hist(x_hist, bins=50, density=True, label=['Test is negative', 'Test is positive'])
plt.legend()
plt.xlabel("Test 2 result")
plt.ylabel("p(value in bin)")
plt.title("p.d.f. of of the test value, given")
plt.show()

# Now, using the defined function:
thresh_list, sens_list, spec_list = findSS(y, x)

plt.figure()
plt.plot(thresh_list, sens_list, 'b', label=r'p($T_p \| D$)')
plt.plot(thresh_list, np.ones((len(spec_list,))) - spec_list, 'r', label=r'p($T_n \| H$)')
plt.grid()
plt.legend()
plt.xlabel('Threshold')
plt.ylabel('Sens / Spec')
plt.title('Sensitivity / Specificity vs. threshold  for Test 2')
plt.show()








##