import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from sub.preprocessor import Preprocessor, buildDataSet, dist_eval
from sub.evaluators import evalAccuracy, evalAccuracyClasses, interCentroidDist, centroidSeparationPlot, plotConfusionMatrix

from sub.preprocessor import sensNames, actNamesShort, actNames

# 19 activities
# 25 Hz sampling frequency
# 8 subjects
# Each activity performed by each subject for 5 minutes
# One file (slice) for each 5 seconds (60 slices)
# Each file contains 125 (= 25 Hz*5 seconds) lines (measurements)

# 5 positions (torso, la, ra ll, rl), 3 sensors per position (acceler., gyro, magnetometers),
# each measuring 3 values (x, y, z) of each parameter
# -> 45 features per measurement

# Each file (slice) contains data in the shape 125(records) x45(sensor signals)

#%%##########################################################################################
# General settings
plt.close('all')

cm = plt.get_cmap('gist_rainbow')
line_styles = ['solid', 'dashed', 'dotted']

filedir = 'data/'

#%%##########################################################################################
# SETTING PARAMETERS
####################
student_ID = 315054
s = student_ID % 8 + 1  # Used subject
print(f"Used subject ID: {s}")

patients = [s]

NAc = 19                                            # Total number of activities

# Total number of sensors
n_sensors_tot = 45
sensors_IDs = list(range(n_sensors_tot))            # List of sensor IDs
sensNamesSub = [sensNames[i] for i in sensors_IDs]  # Names of sensors

# Number of training slices
Nslices = 11
Ntot = 60                                # Total number of slices
slices = list(range(1, Nslices+1))       # First Nslices to plot
print(f"Training set slices: {Nslices}")

fs = 25                                 # Hz, sampling frequency (fixed)
# Samples in each slice (fixed) - each slice is 5 seconds
samplesPerSlice = fs*5

#%%##########################################################################################
# Features to be kept #############################################
# Best comb. of 9 elements:  (OPTIMAL)
used_sensors = [6, 7, 15, 16, 24, 33, 34, 42, 43]

used_sensorNames = [sensNames[i] for i in used_sensors]

""" 
Best accuracy, test: 0.8648421052631579
with elements: (6, 15, 16, 24, 26, 33, 42, 43, 44)
"""

print('Number of used sensors: ', len(used_sensors))
print("Used sensors: ", used_sensorNames)

# Splitting training and test sets
n_slices_tr = Nslices
n_slices_te = Ntot - n_slices_tr

slices_tr = list(range(1, n_slices_tr+1))
slices_te = list(range(n_slices_tr+1, Ntot+1))

n_elem_tr = n_slices_tr*125
n_elem_te = n_slices_te*125

# Consider all activities
activities = list(range(1, NAc + 1))

n_clusters = len(activities)

"""
Preprocessing pipeline: 
- bandstop filter between 0.01 Hz and 11 Hz
- NO UNDERSAMPLING
"""
us_factor = 1
n_elem_tr = round(n_elem_tr/us_factor)
n_elem_te = round(n_elem_te/us_factor)
preproc = Preprocessor(fs=fs, filt_type='bandstop', cutoff=[
                       0.01, 8], us_factor=us_factor)

preproc.plotFreqResp(saveimg=True)

X_tr, y_tr, start_centroids, stdpoints = buildDataSet(filedir, patients, activities,
                                                      slices_tr, used_sensors, preprocessor_obj=preproc, plots=True, savefig=True)


#%%##########################################################################################
# Inter-centroid distance
interCentroidDist(start_centroids, actNamesShort, plot=True, save_img=True)

# Goal: high distance between centroids (clearly a lot of centroids are critically close)
#%%##########################################################################################
# Find minimum distance between each centroid and all others

centroidSeparationPlot(start_centroids, stdpoints,
                       actNamesShort, save_img=True)

#%%##########################################################################################
# K-MEANS ################################

# Initialize centroids as the mean centroids evaluated on the training set
k_means = KMeans(n_clusters=n_clusters, init=start_centroids,
                 n_init=1, max_iter=1000, tol=1e-10)
k_means_fitted = k_means.fit(X_tr)

# Classes id's are between 1 and 19 (not 0 and 18)
# print(k_means_fitted.labels_ + 1)

# Associate to each label the correct activity
mapping_ind = np.zeros((n_clusters,), dtype=np.int8)

# Approach: each centroid corresponds to the class of the
# closest 'centroid' found before as the mean
# (IT WORKS! - Notice: average centroids were also used to initialize K-means centroids)
for i in range(n_clusters):
    centr_curr = k_means_fitted.cluster_centers_[i, :]
    dist_cent = dist_eval(centr_curr, start_centroids)
    mapping_ind[i] = np.argmin(dist_cent)+1

    plt.figure()
    plt.plot(centr_curr, label="centroid "+str(i))
    plt.plot(start_centroids[mapping_ind[i]-1, :],
             ":", label="closest element")
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(X_tr.shape[1]), list(used_sensorNames), rotation=90)
    plt.title("Class "+str(mapping_ind[i]))

if len(np.unique(np.array(mapping_ind))) != 19:
    raise ValueError("Centroids have not been detected correctly!")
print(mapping_ind)
plt.show()

plt.figure()
for i in range(n_clusters):
    lines = plt.plot(k_means_fitted.cluster_centers_[
                     i, :], label=str(mapping_ind[i]))
    lines[0].set_color(cm(i//3*3/NAc))
    lines[0].set_linestyle(line_styles[i % 3])
plt.grid()
plt.legend()
plt.title('Centroids from K-means')
plt.xticks(np.arange(X_tr.shape[1]), list(used_sensorNames), rotation=90)
plt.tight_layout()
plt.savefig('./img/centroids.png')
plt.show()

# Test set

X_te, y_te = buildDataSet(filedir, patients, activities, slices_te,
                          used_sensors, preprocessor_obj=preproc, plots=False)[:2]

# Performance analysis

y_hat_tr = k_means_fitted.predict(X_tr)
y_hat_te = k_means_fitted.predict(X_te)

# Accuracy
acc_tr_kmeans = evalAccuracy(y_hat_tr+1, y_tr)
acc_te_kmeans = evalAccuracy(y_hat_te+1, y_te)
# Notice: the label saved in y_hat goes from 0 to 18, not from 1 to 19

# Confusion matrix
cm_tr = plotConfusionMatrix(y_hat_tr+1, y_tr, actNamesShort,
                            title='Confusion Matrix, train', save_img=True, img_path='img/conf_mat_tr.png')
cm_te = plotConfusionMatrix(y_hat_te+1, y_te, actNamesShort,
                            title='Confusion Matrix, test', save_img=True, img_path='img/conf_mat_te.png')

print(f"Accuracy (training): {acc_tr_kmeans}")
print(f"Accuracy (test): {acc_te_kmeans}")

# Accuracies for each activity
acc_act_tr = evalAccuracyClasses(y_hat_tr + 1, y_tr)
acc_act_te = evalAccuracyClasses(y_hat_te + 1, y_te)

acc_class_nd = np.stack(
    [acc_act_tr.reshape((NAc,)), acc_act_te.reshape((NAc,))], axis=0)

acc_classes = pd.DataFrame(acc_class_nd, columns=actNamesShort, index=[
                           'Accuracy, train', 'Accuracy, test'])

pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 150)

print(acc_classes)
