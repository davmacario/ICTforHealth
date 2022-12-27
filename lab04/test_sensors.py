import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sub.preprocessor import generateDF, preprocessor, buildDataSet, dist_eval
from sklearn.cluster import KMeans
from sub.evaluators import evalAccuracy, interCentroidDist, centroidSeparationPlot

import itertools as it
import copy

from sub.preprocessor import sensNames, actNames, actNamesShort

from multiprocessing import Pool

""" 
- TODO[7]: Plot confusion matrix for both training and test sets
- TODO[9]: Improve system for accuracy ~= 85%-90%
"""


def run_tries_loop(operation, input, pool):
    pool.map(operation, input)

def tries_loop(sens_comb):
    best_acc_tr = 0
    best_acc_te = 0

    set_best_acc_tr = []
    set_best_acc_te = []

    for used_sensors in sens_comb:
        used_sensorNames = [sensNames[i] for i in used_sensors]

        print('Number of used sensors: ', len(used_sensors))

        # Features to be replaced with variance at undersampling #############
        takevar_ind = []

        takevar_names = [sensNames[i] for i in takevar_ind]

        ####
        n_slices_tr = Nslices
        n_slices_te = Ntot - n_slices_tr

        slices_tr = list(range(1, n_slices_tr+1))
        slices_te = list(range(n_slices_tr+1, Ntot+1))

        # Consider all activities
        activities = list(range(1, NAc + 1))

        n_clusters = len(activities)

        X_tr, y_tr, start_centroids, stdpoints = buildDataSet(filedir, patients, activities, slices_tr, used_sensors, used_sensorNames, takevar_ind, ID='train', plots=False)

        #%%##########################################################################################
        # Inter-centroid distance
        interCentroidDist(start_centroids, actNamesShort, plot=False, save_img=True)

        # Goal: high distance between centroids (clearly a lot of centroids are critically close)
        #%%##########################################################################################
        # Find minimum distance between each centroid and all others

        #centroidSeparationPlot(start_centroids, stdpoints, actNamesShort, save_img=True)


        # K-means - initialize centroids as the mean centroids evaluated on the training set
        k_means = KMeans(n_clusters=n_clusters, init=start_centroids, n_init=1, max_iter=1000, tol=1e-10)
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

        #if len(np.unique(np.array(mapping_ind))) != 19:
            #raise ValueError("Centroids have not been detected correctly!")

        X_te, y_te = buildDataSet(filedir, patients, activities, slices_te, used_sensors, used_sensorNames, takevar_ind, ID='test', plots=False)[:2]

        y_hat_tr = k_means_fitted.predict(X_tr)
        y_hat_te = k_means_fitted.predict(X_te)

        # Accuracy
        acc_tr_kmeans = evalAccuracy(y_hat_tr+1, y_tr)
        acc_te_kmeans = evalAccuracy(y_hat_te+1, y_te)
        #### Notice: the label saved in y_hat_te goes from 0 to 18, not from 1 to 19

        if acc_tr_kmeans > best_acc_tr:
            best_acc_tr = copy.copy(acc_tr_kmeans)
            set_best_acc_tr = used_sensors

        if acc_te_kmeans > best_acc_te:
            best_acc_te = copy.copy(acc_te_kmeans)
            set_best_acc_te = used_sensors

    print(f"Best accuracy, training: {best_acc_tr}")
    print(f"with elements: {set_best_acc_tr}")
    print("")
    print(f"Best accuracy, test: {best_acc_te}")
    print(f"with elements: {set_best_acc_te}")

    return (best_acc_te, set_best_acc_te)

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

sensors_combinations = list()

acc_id = [6, 7, 15, 16, 24, 33, 34, 35, 42, 43]
gyro_id = [n + 3 for n in acc_id]
mag_id = [n + 6 for n in acc_id]

valid_sens = [6, 7, 15, 16, 24, 33, 34, 35, 42, 43]

for n in [9]:
    sensors_combinations += list(it.combinations(valid_sens, n))
    #sensors_combinations += list(it.combinations(sensors_IDs, n))

# Acc_te to beat: 0.8551578947368421

best_acc_te, set_best_acc_te = tries_loop(sensors_combinations)
