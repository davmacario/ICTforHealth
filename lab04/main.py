import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sub.preprocessor import generateDF, preprocessor, buildDataSet
from sklearn.cluster import KMeans
from sub.evaluators import evalAccuracy

""" 
- TODO[7]: Plot confusion matrix for both training and test sets
- TODO[8]: Find accuracy for both sets (uniform prior)
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




def dist_eval(element, train):
    """
    dist_eval: evaluate the distance (euclidean sense) between the test element
    and each one of the elements of the training set
    ------------------------------------------------------------------------------
    - element: item whose distance needs to be computed
    - train: training set; each row is an element and the columns are the features
    ------------------------------------------------------------------------------
    """
    # Check same n. of features
    if element.shape[0] != train.shape[1]:
        raise ValueError(
            'Error! The number of features of the element is not the consistent!')

    distance_vect = np.empty((train.shape[0],))

    for ind2 in range(train.shape[0]):
        tmp_sum = sum(np.power(element - train[ind2, :], 2))
        distance_vect[ind2] = np.sqrt(tmp_sum)

    return distance_vect

############################################################################################


plt.close('all')
cm = plt.get_cmap('gist_rainbow')
line_styles = ['solid', 'dashed', 'dotted']

filedir1 = 'data/'
filedir2 = 'lab04/data/'

sensNames = [
    'T_xacc', 'T_yacc', 'T_zacc',
    'T_xgyro', 'T_ygyro', 'T_zgyro',
    'T_xmag', 'T_ymag', 'T_zmag',
    'RA_xacc', 'RA_yacc', 'RA_zacc',
    'RA_xgyro', 'RA_ygyro', 'RA_zgyro',
    'RA_xmag', 'RA_ymag', 'RA_zmag',
    'LA_xacc', 'LA_yacc', 'LA_zacc',
    'LA_xgyro', 'LA_ygyro', 'LA_zgyro',
    'LA_xmag', 'LA_ymag', 'LA_zmag',
    'RL_xacc', 'RL_yacc', 'RL_zacc',
    'RL_xgyro', 'RL_ygyro', 'RL_zgyro',
    'RL_xmag', 'RL_ymag', 'RL_zmag',
    'LL_xacc', 'LL_yacc', 'LL_zacc',
    'LL_xgyro', 'LL_ygyro', 'LL_zgyro',
    'LL_xmag', 'LL_ymag', 'LL_zmag']

actNames = [
    'sitting',  # 1
    'standing',  # 2
    'lying on back',  # 3
    'lying on right side',  # 4
    'ascending stairs',  # 5
    'descending stairs',  # 6
    'standing in an elevator still',  # 7
    'moving around in an elevator',  # 8
    'walking in a parking lot',  # 9
    'walking on a treadmill with a speed of 4 km/h in flat',  # 10
    'walking on a treadmill with a speed of 4 km/h in 15 deg inclined position',  # 11
    'running on a treadmill with a speed of 8 km/h',  # 12
    'exercising on a stepper',  # 13
    'exercising on a cross trainer',  # 14
    'cycling on an exercise bike in horizontal positions',  # 15
    'cycling on an exercise bike in vertical positions',  # 16
    'rowing',  # 17
    'jumping',  # 18
    'playing basketball'  # 19
]

actNamesShort = [
    'sitting',  # 1
    'standing',  # 2
    'lying.ba',  # 3
    'lying.ri',  # 4
    'asc.sta',  # 5
    'desc.sta',  # 6
    'stand.elev',  # 7
    'mov.elev',  # 8
    'walk.park',  # 9
    'walk.4.fl',  # 10
    'walk.4.15',  # 11
    'run.8',  # 12
    'exer.step',  # 13
    'exer.train',  # 14
    'cycl.hor',  # 15
    'cycl.ver',  # 16
    'rowing',  # 17
    'jumping',  # 18
    'play.bb'  # 19
]


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
# Plot centroids and stand. dev. of sensor values
# All activities considered, but only 9 features

# tbr_names = ['T_xgyro','T_ygyro','T_zgyro',
#     'RA_xgyro','RA_ygyro','RA_zgyro',
#     'LA_xgyro','LA_ygyro','LA_zgyro',
#     'RL_xgyro','RL_ygyro','RL_zgyro',
#     'LL_xgyro','LL_ygyro','LL_zgyro']

# Only keep Accel
# tbr_ind = [3, 4, 5, 6, 7, 8,
#             12, 13, 14, 15, 16, 17,
#             21, 22, 23, 24, 25, 26,
#             30, 31, 32, 33, 34, 35,
#             39, 40, 41, 42, 43, 44]

# Only keep Gyro
# tbr_ind = [0, 1, 2, 6, 7, 8,
#             9, 10, 11, 15, 16, 17,
#             18, 19, 20, 24, 25, 26,
#             27, 28, 29, 33, 34, 35,
#             36, 37, 38, 42, 43, 44]

# Only keep Magnetometer               # Best for now
# tbr_ind = [0, 1, 2, 3, 4, 5,
#            9, 10, 11, 12, 13, 14,
#            18, 19, 20, 21, 22, 23,
#            27, 28, 29, 30, 31, 32,
#            36, 37, 38, 39, 40, 41]

# Keep accelerometer and magnetometer
# tbr_ind = [0, 2, 3, 4, 5,
#             9, 11, 12, 13, 14,
#             18, 20, 21, 22, 23,
#             27, 28, 30, 31, 32,
#             36, 37, 39, 40, 41]


# Keep magnetometer plus legs accel.
# tbr_ind = [0, 1, 2, 3, 4, 5,
#             9, 10, 11, 12, 13, 14,
#             18, 19, 20, 21, 22, 23,
#             27, 28, 29, 30, 31, 32,
#             36, 37, 38, 39, 40, 41]


#
#
#


# Features to be removed #############################################
tbr_ind = [0, 1, 2, 3, 4, 5,
           9, 10, 11, 12, 13, 14,
           18, 19, 20, 21, 22, 23,
           27, 28, 29, 30, 31, 32,
           36, 37, 38, 39, 40, 41]

# Features to be averaged - MSV ######################################
tba_ind = [[]]

# Features to be replaced with variance at undersampling #############
takevar_ind = []

######################################################################
# Translate into strings:
tbr_names = [sensNames[i] for i in tbr_ind]
tba_names = []
if (len(tba_ind[0]) > 0):  # !!! tba_ind = [[]] has length 1...
    for i in range(len(tba_ind)):
        tba_names.append([sensNames[j] for j in tba_ind[i]])

takevar_names = [sensNames[i] for i in takevar_ind]

######################################################################

used_sensors = [elem for elem in sensors_IDs if elem not in tbr_ind]
used_sensorNames = [sensNames[i] for i in used_sensors]

print('Number of used sensors: ', len(used_sensors))

# # Centroids for all the activities
# centroids = np.zeros(
#     (NAc, len(used_sensors) - sum(len(i)-1 for i in tba_names)))
# # Variance in cluster for each sensor
# stdpoints = np.zeros(
#     (NAc, len(used_sensors) - sum(len(i)-1 for i in tba_names)))

# plt.figure(figsize=(12, 6))

# for i in range(1, NAc + 1):         # Extract all activities
#     activities = [i]
#     try:
#         x = generateDF(filedir1, sensNamesSub, sensors_IDs, patients, activities, slices)
#     except:
#         x = generateDF(filedir2, sensNamesSub, sensors_IDs, patients, activities, slices)

#     x = x.drop(columns=['activity'])

#     # PREPROCESSING
#     # Drop features
#     # Undersampling - consider as samples the average of 25 measurements - from 25 Hz to 1
#     # DBSCAN
#     x = preprocessor(x, drop_feat=tbr_names, us_factor=2*fs, dbscan=True,
#                      dbscan_eps=1.2, dbscan_M=6, takeVar=takevar_names,
#                      var_norm=True, var_thresh=1) # (Nslices*125)x(n_sensors)

#     # Note indexing - centroid in position k corresponds to class k+1
#     centroids[i-1, :] = x.mean().values
#     stdpoints[i-1] = np.sqrt(x.var().values)        # Update stdev

#     plt.subplot(1, 2, 1)
#     lines = plt.plot(centroids[i-1, :], label=actNamesShort[i-1])
#     lines[0].set_color(cm(i//3*3/NAc))
#     lines[0].set_linestyle(line_styles[i % 3])

#     plt.subplot(1, 2, 2)
#     lines = plt.plot(stdpoints[i-1, :], label=actNamesShort[i-1])
#     lines[0].set_color(cm(i//3*3/NAc))
#     lines[0].set_linestyle(line_styles[i % 3])

# plt.subplot(1, 2, 1)
# plt.legend(loc='upper right')
# plt.grid()
# plt.title('Centroids using '+str(len(used_sensors))+' sensors')
# plt.xticks(np.arange(x.shape[1]), list(x.columns), rotation=90)

# plt.subplot(1, 2, 2)
# plt.legend(loc='upper right')
# plt.grid()
# plt.title('Standard deviation using '+str(len(used_sensors))+' sensors')
# plt.xticks(np.arange(x.shape[1]), list(x.columns), rotation=90)

# plt.tight_layout()
# plt.show()






# exit()


#%%################# Classification #################################

### TODO: Insert here variables used (lists of names/slices/...)

n_slices_tr = Nslices
n_slices_te = Ntot - n_slices_tr

slices_tr = list(range(1, n_slices_tr+1))
slices_te = list(range(n_slices_tr+1, Ntot+1))

# Consider all activities
activities = list(range(1, NAc + 1))

n_clusters = len(activities)

X_tr, y_tr, start_centroids, stdpoints = buildDataSet(filedir1, patients, activities, slices_tr, sensors_IDs, sensNamesSub, tbr_names, tba_ind, ID='train', plots=True)


#%%##########################################################################################
# Inter-centroid distance
# Element i, j is squared norm of dist between centroid i and j
d = np.zeros((NAc, NAc))

for i in range(NAc):
    for j in range(i+1, NAc):       # Optimization - diagonal elements are 0 and matrix is symmetric
        d[i, j] = np.linalg.norm(start_centroids[i] - start_centroids[j])
        d[j, i] = d[i, j]

plt.matshow(d)
plt.colorbar()
plt.xticks(np.arange(NAc), actNamesShort, rotation=90)
plt.yticks(np.arange(NAc), actNamesShort)
plt.title('Between-centroids distance')
plt.show()

# Goal: high distance between centroids (clearly a lot of centroids are critically close)
#%%##########################################################################################
# Find minimum distance between each centroid and all others

# Remove zeros on the diagonal (distance of centroid from itself)
dd = d+np.eye(NAc)*1e6

dmin = dd.min(axis=0)         # Find the minimum distance for each centroid

# Average distance between each centroid and its points 
dpoints = np.sqrt(np.sum(stdpoints**2, axis=1))

plt.figure()
plt.plot(dmin, label='minimum centroid distance')
plt.plot(dpoints, label='mean distance from points to centroid')
plt.grid()
plt.xticks(np.arange(NAc), actNamesShort, rotation=90)
plt.legend()
plt.tight_layout()
# if the minimum distance is less than the mean distance, then some points of the cluster are closer
# to another centroid
plt.show()


# K-means - initialize centroids as 
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

X_te, y_te = buildDataSet(filedir1, patients, activities, slices_te, sensors_IDs, sensNamesSub, tbr_names, tba_ind, ID='test', plots=False)[:2]

y_hat_tr = k_means_fitted.predict(X_tr)
y_hat_te = k_means_fitted.predict(X_te)

# Accuracy
acc_tr_kmeans = evalAccuracy(y_hat_tr+1, y_tr)
acc_te_kmeans = evalAccuracy(y_hat_te+1, y_te)
#### Notice: the label saved in y_hat_te goes from 0 to 18, not from 1 to 19

print(f"Accuracy (training): {acc_tr_kmeans}")
print(f"Accuracy (test): {acc_te_kmeans}")
