import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sub.preprocessor import generateDF, preprocessor, buildDataSet, dist_eval
from sklearn.cluster import KMeans
from sub.evaluators import evalAccuracy, interCentroidDist, centroidSeparationPlot

from sub.preprocessor import sensNames, actNames, actNamesShort

""" 
- TODO[7]: Plot confusion matrix for both training and test sets
- TODO[9]: Improve system for accuracy ~= 85%-90%
"""


# 19 activities
# 25 Hz sampling frequency
# 8 subjects
# Each activity performed by each subject for 5 minutes
# One file for each 5 seconds (60 files)
# Each file contains 125 (= 25 Hz*5 seconds) lines (measurements)

# 5 positions (torso, la, ra ll, rl), 3 sensors per position (acceler., gyro, magnetometers),
# each measuring 3 values (x, y, z) of each parameter
# -> 45 features per measurement

# Each file contains data in the shape 125x45

#%%##########################################################################################
plt.close('all')

cm = plt.get_cmap('gist_rainbow')
line_styles = ['solid', 'dashed', 'dotted']

filedir = 'data/'

#%%##########################################################################################

student_ID = 315054
s = student_ID % 8 + 1  # Used subject
print(f"Used subject: {s}")

patients = [s]

NAc = 19                                # Total number of activities

# Total number of sensors (TO BE TUNED)
n_sensors_tot = 45
sensors_IDs = list(range(n_sensors_tot))        # List of sensors
sensNamesSub = [sensNames[i] for i in sensors_IDs]  # Names of selected sensors

# Number of slices to plot (TO BE TUNED)
Nslices = 10
# Nslices = 60
Ntot = 60                               # Total number of slices
slices = list(range(1, Nslices+1))       # First Nslices to plot

fs = 25                                 # Hz, sampling frequency (fixed)
samplesPerSlice = fs*5                  # Samples in each slice (fixed) - each slice is 5 seconds


#%%##########################################################################################
# Features to be kept #############################################
# used_sensors = [6, 7, 8, 15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44]

# Evaluated as the best combination of 12 elements in terms of accuracy on the test set
# used_sensors = [6, 15, 16, 17, 24, 26, 33, 34, 35, 42, 43, 44]

# Best comb. of 9 elements:
used_sensors = [6, 15, 16, 24, 26, 33, 42, 43, 44]

used_sensorNames = [sensNames[i] for i in used_sensors]

""" 
Best accuracy, test: 0.8648421052631579
with elements: (6, 15, 16, 24, 26, 33, 42, 43, 44)
"""

print('Number of used sensors: ', len(used_sensors))

# Features to be averaged - MSV ######################################
# (must be included in the used_sensors list)
tba_ind = [[]]

# Features to be replaced with variance at undersampling #############
takevar_ind = [33, 34, 35, 42, 43, 44]

# Translate into strings:
tba_names = []
if (len(tba_ind[0]) > 0):  # !!! tba_ind = [[]] has length 1...
    for i in range(len(tba_ind)):
        tba_names.append([sensNames[j] for j in tba_ind[i]])

takevar_names = [sensNames[i] for i in takevar_ind]

####
n_slices_tr = Nslices
n_slices_te = Ntot - n_slices_tr

slices_tr = list(range(1, n_slices_tr+1))
slices_te = list(range(n_slices_tr+1, Ntot+1))

# Consider all activities
activities = list(range(1, NAc + 1))

n_clusters = len(activities)

X_tr, y_tr, start_centroids, stdpoints = buildDataSet(filedir, patients, activities,\
         slices_tr, used_sensors, used_sensorNames, takevar_names, ID='train', plots=True)

#%%##########################################################################################
# Inter-centroid distance
interCentroidDist(start_centroids, actNamesShort, plot=True, save_img=True)

# Goal: high distance between centroids (clearly a lot of centroids are critically close)
#%%##########################################################################################
# Find minimum distance between each centroid and all others

centroidSeparationPlot(start_centroids, stdpoints, actNamesShort, save_img=True)


# K-means - initialize centroids as the mean centroids evaluated on the training set
k_means = KMeans(n_clusters=n_clusters, init=start_centroids, max_iter=1000, tol=1e-10)
k_means_fitted = k_means.fit(X_tr)

# Classes id's are between 1 and 19 (not 0 and 18)
# print(k_means_fitted.labels_ + 1)

# Associate to each label the correct activity
mapping_ind = np.zeros((n_clusters,), dtype=np.int8)

# Approach (2): each centroid corresponds to the class of the 
# closest 'centroid' found before as the mean 
# (IT WORKS! - Notice: average centroids were also used to initialize K-means centroids)
for i in range(n_clusters):
    centr_curr = k_means_fitted.cluster_centers_[i, :]
    dist_cent = dist_eval(centr_curr, start_centroids)
    mapping_ind[i] = np.argmin(dist_cent)+1
    
    plt.figure()
    plt.plot(centr_curr, label="centroid "+str(i))
    plt.plot(start_centroids[mapping_ind[i]-1, :], ":", label="closest element")
    plt.grid()
    plt.legend()
    plt.title("Class "+str(mapping_ind[i]))

if len(np.unique(np.array(mapping_ind))) != 19:
    raise ValueError("Centroids have not been detected correctly!")
print(mapping_ind)
plt.show()

plt.figure()
for i in range(n_clusters):
    lines = plt.plot(k_means_fitted.cluster_centers_[i, :], label=str(mapping_ind[i]))
    lines[0].set_color(cm(i//3*3/NAc))
    lines[0].set_linestyle(line_styles[i % 3])
plt.grid()
plt.legend()
plt.title('Centroids from K-means')
plt.xticks(np.arange(X_tr.shape[1]), list(used_sensorNames), rotation=90)
plt.show()

X_te, y_te = buildDataSet(filedir, patients, activities, slices_te, used_sensors, used_sensorNames, takevar_names, ID='test', plots=False)[:2]

y_hat_tr = k_means_fitted.predict(X_tr)
y_hat_te = k_means_fitted.predict(X_te)

# Accuracy
acc_tr_kmeans = evalAccuracy(y_hat_tr+1, y_tr)
acc_te_kmeans = evalAccuracy(y_hat_te+1, y_te)
#### Notice: the label saved in y_hat_te goes from 0 to 18, not from 1 to 19

print(f"Accuracy (training): {acc_tr_kmeans}")
print(f"Accuracy (test): {acc_te_kmeans}")
