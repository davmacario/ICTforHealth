import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk


def findSS(y, x):
    """
    - x: value of the test (on which to apply threshold)
    - y: actual correct outcome of the test
    """
    # Sort values of x and use them as thresholds
    ind_sort = np.argsort(x)
    thresh = np.append([0], x[ind_sort])    # Notice: added 0

    sens_list = []
    spec_list = []

    x0 = x[y == 0]  # values of 'x' associated with negative outcome 
    x1 = x[y == 1]  # values of 'x' associated with positive outcome 

    Np = np.sum(y == 1)    # number of ill
    Nn = np.sum(y == 0)    # number of healthy

    for i in range(len(thresh)):
        thresh_now = thresh[i]

        # Sensitivity: number of correctly detected positives over total number of ill
        n1 = np.sum(x1 > thresh_now)
        sens_list.append(n1/Np)

        # Sensitivity: number of correctly detected negatives over total number of healthy
        n0 = np.sum(x0 < thresh_now)
        spec_list.append(n0/Nn)

    sensitivity = np.array(sens_list)
    specificity = np.array(spec_list)

    return thresh, sensitivity, specificity


def myTrapezoidalRule(y, x):
    tmp_int = 0

    for i in range(len(x)-1):
        tmp_int += 0.5*(y[i] + y[i+1])*(x[i+1] - x[i])

    return tmp_int


plt.rcParams["font.family"] = "Times New Roman"
# %%
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
swab = xx.COVID_swab_res.values
swab[swab >= 1] = 1  # non reliable values are considered positive

Test1 = xx.IgG_Test1_titre.values
Test2 = xx.IgG_Test2_titre.values

# Plot test 2 vs. test 1 - see outliers
plt.figure()
plt.plot(Test1[swab == 0], Test2[swab == 0], '.', label='neg')
plt.plot(Test1[swab == 1], Test2[swab == 1], '.', label='pos')
plt.title("Visualize outliers")
plt.xlabel("Test 1")
plt.ylabel("Test 2")
plt.legend()
plt.grid()
plt.tight_layout()
try:
    plt.savefig('./img/outliers_visual.png')
except:
    plt.savefig('./lab03/img/outliers_visual.png')
# plt.show()

data = xx.values
N_points = data.shape[0]

data_norm = (data - data.mean())/data.std()       # Normalize data

# Carry out DBSCAN, setting suitable parameters (eps, M)
clustering = sk.DBSCAN(eps=0.5, min_samples=5).fit(data_norm)

ii = np.argwhere(clustering.labels_ == -1)[:, 0]    # Outliers
# Print outlier(s)
print("Detected outliers: ")
print(xx.iloc[ii])

# Drop the outliers
xx = xx.drop(ii)

# Re-extract data:
swab = xx.COVID_swab_res.values         # 0 = no illness; 1 = illness
Test1 = xx.IgG_Test1_titre.values
Test2 = xx.IgG_Test2_titre.values

############# Sensitivity and Specificity #######################################

############# Test1 #################################################################
x = Test1
y = swab

x0 = x[swab == 0]     # Test results for healthy
x1 = x[swab == 1]     # Test results for ill

Np = np.sum(swab == 1)    # number of ill
Nn = np.sum(swab == 0)    # number of healthy

# Set the thresh
thresh = 5

n1 = np.sum(x1 > thresh)
sens = n1/Np

n0 = np.sum(x0 < thresh)
spec = n0/Nn

print("-------------------------------------------------------------")
print(f"Results - threshold = {thresh}")
print(f"Sensitivity - test 1: {sens}\nSpecificity - test 1: {spec}")
print("-------------------------------------------------------------")

plt.figure()
x_hist = [x0, x1]
plt.hist(x_hist, bins=50, density=True, label=[
         r'H', r'D'])
plt.legend()
plt.xlabel("Test 1 value")
plt.ylabel("p(value in bin)")
plt.title("p.d.f. of of the test value, given swab test result")
plt.grid()
plt.tight_layout()
try:
    plt.savefig('./img/1_pdf_hist.png')
except:
    plt.savefig('./lab03/img/1_pdf_hist.png')
# plt.show()

# Now, using the defined function:
thresh_list_1, sens_list_1, spec_list_1 = findSS(y, x)

plt.figure()
plt.plot(thresh_list_1, sens_list_1, 'b', label=r'p($T_p \| D$)')
plt.plot(thresh_list_1, spec_list_1, 'r', label=r'p($T_n \| H$)')
plt.grid()
plt.legend()
plt.xlabel('Threshold')
plt.ylabel('Sens / Spec')
plt.title('Sensitivity / Specificity vs. threshold  for Test 1')
plt.tight_layout()
try:
    plt.savefig('./img/1_sens_spec_thresh.png')
except:
    plt.savefig('./lab03/img/1_sens_spec_thresh.png')
# plt.show()

# ROC curve
FA_1 = 1 - spec_list_1

plt.figure()
plt.plot(FA_1, sens_list_1, 'b')
plt.plot([0, 1], [1, 0], 'r:', linewidth=0.5)
plt.grid()
plt.xlabel(r"$p(T_p \| H)$")
plt.ylabel(r"$p(T_p \| D)$")
plt.title("ROC curve - test 1")
plt.tight_layout()
try:
    plt.savefig('./img/1_ROC.png')
except:
    plt.savefig('./lab03/img/1_ROC.png')
# plt.show()

# AUC
# Integrate via trapezoidal rule
AuC = myTrapezoidalRule(sens_list_1[::-1], FA_1[::-1])
print(f"AuC - test 1: {AuC}")
print("-------------------------------------------------------------")

prevalence = 0.01

npv_1 = (spec_list_1*(1-prevalence)) / \
    ((1-sens_list_1)*prevalence + spec_list_1*(1-prevalence))

p_ill_given_neg_1 = 1 - npv_1

ppv_1 = (sens_list_1*prevalence) / \
    (sens_list_1*prevalence + FA_1*(1-prevalence))

p_h_given_pos_1 = 1 - ppv_1

plt.figure()
plt.plot(thresh_list_1, p_ill_given_neg_1, label=r'$p(D|T_n)$')
plt.plot(thresh_list_1, ppv_1, label=r'$p(D|T_p)$')
plt.legend()
plt.grid()
plt.xlabel('threshold')
plt.ylabel('Probability')
plt.title(r"$p(D|T_n)$ and precision comparison - test 1")
plt.tight_layout()
try:
    plt.savefig('./img/1_precision_comparison.png')
except:
    plt.savefig('./lab03/img/1_precision_comparison.png')
# plt.show()

plt.figure()
plt.plot(thresh_list_1, npv_1, label=r'$p(H|T_n)$')
plt.plot(thresh_list_1, p_h_given_pos_1, label=r'$p(H|T_p)$')
plt.legend()
plt.grid()
plt.xlabel('threshold')
plt.ylabel('Probability')
plt.title(r"Neg. predictive value and $p(H|T_p)$ comparison - test 1")
plt.tight_layout()
try:
    plt.savefig('./img/1_npv_comparison.png')
except:
    plt.savefig('./lab03/img/1_npv_comparison.png')
# plt.show()

plt.figure()
plt.plot(ppv_1, npv_1)
plt.grid()
plt.ylabel(r"p(H|T_n)")
plt.xlabel(r"p(D|T_p)")
plt.title(r"Test 1")
plt.tight_layout()
try:
    plt.savefig('./img/1_npv_vs_ppv.png')
except:
    plt.savefig('./lab03/img/1_npv_vs_ppv.png')
# plt.show()


############# Test2 ##################################################################
x = Test2

x0 = x[swab == 0]     # Test results for healthy
x1 = x[swab == 1]     # Test results for ill

Np = np.sum(swab == 1)    # number of ill
Nn = np.sum(swab == 0)    # number of healthy

# Set the thresh
thresh = 5

n1 = np.sum(x1 > thresh)
sens = n1/Np

n0 = np.sum(x0 < thresh)
spec = n0/Nn

print("-------------------------------------------------------------")
print(f"Results - threshold = {thresh}")
print(f"Sensitivity - test 2: {sens}\nSpecificity - test 2: {spec}")
print("-------------------------------------------------------------")

plt.figure()
x_hist = [x0, x1]
plt.hist(x_hist, bins=50, density=True, label=[
         r'$H$', r'D'])
plt.legend()
plt.xlabel("Test 2 value")
plt.ylabel("p(value in bin)")
plt.title("p.d.f. of of the test value, given swab test result")
plt.tight_layout()
try:
    plt.savefig('./img/2_pdf_hist.png')
except:
    plt.savefig('./lab03/img/2_pdf_hist.png')
# plt.show()

# Now, using the defined function:
thresh_list_2, sens_list_2, spec_list_2 = findSS(y, x)

plt.figure()
plt.plot(thresh_list_2, sens_list_2, 'b', label=r'p($T_p \| D$)')
plt.plot(thresh_list_2, spec_list_2, 'r', label=r'p($T_n \| H$)')
plt.grid()
plt.legend()
plt.xlabel('Threshold')
plt.ylabel('Sens / Spec')
plt.title('Sensitivity / Specificity vs. threshold  for Test 2')
plt.tight_layout()
try:
    plt.savefig('./img/2_sens_spec_thresh.png')
except:
    plt.savefig('./lab03/img/2_sens_spec_thresh.png')
# plt.show()

# ROC curve
FA_2 = 1 - spec_list_2

plt.figure()
plt.plot(FA_2, sens_list_2, 'b')
plt.plot([0, 1], [1, 0], 'r:', linewidth=0.5)
plt.grid()
plt.xlabel(r"$p(T_p \| H)$")
plt.ylabel(r"$p(T_p \| D)$")
plt.title("ROC curve - test 2")
plt.tight_layout()
try:
    plt.savefig('./img/2_ROC.png')
except:
    plt.savefig('./lab03/img/2_ROC.png')
# plt.show()

# AUC
# Integrate via trapezoidal rule
AuC = myTrapezoidalRule(sens_list_2[::-1], FA_2[::-1])
print(f"AuC - test 2: {AuC}")
print("-------------------------------------------------------------")


# ROC comparisons

plt.figure()
plt.plot(FA_1, sens_list_1, label='test 1')
plt.plot(FA_2, sens_list_2, label='test 2')
plt.plot([0, 1], [1, 0], 'r:', linewidth=0.5)
plt.grid()
plt.legend()
plt.xlabel(r"$p(T_p \| H)$")
plt.ylabel(r"$p(T_p \| D)$")
plt.title("ROC curve - comparison")
plt.tight_layout()
try:
    plt.savefig('./img/ROC_comparison.png')
except:
    plt.savefig('./lab03/img/ROC_comparison.png')
# plt.show()

prevalence = 0.01

npv_2 = (spec_list_2*(1-prevalence)) / \
    ((1-sens_list_2)*prevalence + spec_list_2*(1-prevalence))

p_ill_given_neg_2 = 1 - npv_2

ppv_2 = (sens_list_2*prevalence) / \
    (sens_list_2*prevalence + FA_2*(1-prevalence))

p_h_given_pos_2 = 1 - ppv_2


plt.figure()
plt.plot(thresh_list_2, p_ill_given_neg_2, label=r'$p(D|T_n)$')
plt.plot(thresh_list_2, ppv_2, label=r'$p(D|T_p)$')
plt.legend()
plt.grid()
plt.xlabel('threshold')
plt.ylabel('Probability')
plt.title(r"$p(D|T_n)$ and precision comparison - test 2")
plt.tight_layout()
try:
    plt.savefig('./img/2_precision_comparison.png')
except:
    plt.savefig('./lab03/img/2_precision_comparison.png')
# plt.show()

plt.figure()
plt.plot(thresh_list_2, npv_2, label=r'$p(H|T_n)$')
plt.plot(thresh_list_2, p_h_given_pos_2, label=r'$p(H|T_p)$')
plt.legend()
plt.grid()
plt.xlabel('threshold')
plt.ylabel('Probability')
plt.title(r"Negative predictive value and $p(H|T_p)$ comparison - test 2")
plt.tight_layout()
try:
    plt.savefig('./img/2_npv_comparison.png')
except:
    plt.savefig('./lab03/img/2_npv_comparison.png')
# plt.show()

plt.figure()
plt.plot(ppv_2, npv_2)
plt.grid()
plt.ylabel(r"p(H|T_n)")
plt.xlabel(r"p(D|T_p)")
plt.title(r"Test 2")
plt.tight_layout()
try:
    plt.savefig('./img/2_npv_vs_ppv.png')
except:
    plt.savefig('./lab03/img/2_npv_vs_ppv.png')
# plt.show()
