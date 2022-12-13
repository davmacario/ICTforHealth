import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk

def generateDF(filedir, colnames, sensors, patients, activities, slices):
    """
    generateDF
    --------------------------------------------------------------
    Get the data from files for the selected patients and selected
    activities.
    Then concatenate all the slices and generate a pandas 
    dataframe with an added column: activity 
    --------------------------------------------------------------
    Parameters:
    - filedir: (string) path of the file
    - colnames: (list of strings) selected columns to be extracted
      (corresponding to the used sensors)
    - sensors: (list of ints) ID of the sensors in 'colnames'
    - patients: (list of ints) list of considered patients
    - activities: (list of ints) list of considered activities 
      (IDs)
    - slices: (list of ints) list of considered slices (IDs)
    --------------------------------------------------------------
    """
    x=pd.DataFrame()
    for pat in patients:
        for a in activities:
            subdir='a'+f"{a:02d}"+'/p'+str(pat)+'/'
            for s in slices:
                filename=filedir+subdir+'s'+f"{s:02d}"+'.txt'
                #print(filename)
                x1=pd.read_csv(filename,usecols=sensors,names=colnames)
                x1['activity']=a*np.ones((x1.shape[0],),dtype=int)
                x=pd.concat([x,x1], axis=0, join='outer', ignore_index=True, 
                            keys=None, levels=None, names=None, verify_integrity=False, 
                            sort=False, copy=True)
    return x

def preprocessor(df, drop_feat=[], us_factor=1, takeVar=[], dbscan=False, dbscan_eps=1, dbscan_M=5, msv_list=[[]], var_norm=False, var_thresh=1):
    """
    preprocessor
    --------------------------------------------------------------
    Apply a preprocessing pipeline to an input dataframe.

    Possible preprocessing strategies:
    - Drop features
    - Undersampling
        - When downsampling, replace feature with variance
    - Mean squared value of x, y, z of each measure
    - Normalize variance of selected features
    --------------------------------------------------------------
    Input parameters:
    - df: input dataframe
    - drop_feat: list of features of the dataframe to be dropped
    - us_factor: undersampling factor
    - takeVar: flag for substituting average feature with 
      variance (WHAT IF VAR = 0?)
    - dbscan: flag for performing dbscan
    - dbscan_eps: hypersphere radius of DBSCAN
    - dbscan_M: number of neighbors dbscan
    - msv_list = list of lists including the features to be 
      replaced with their mean square value (after undersampling)

    - var_norm: flag for performing variance normalization
    - var_eps: value of the variance above which normalization is 
      performed
    --------------------------------------------------------------
    """

    df_start = df.copy()
    n_p, n_f = df.shape

    # Drop features
    df_start = df_start.drop(columns=drop_feat)

    feat_list = df_start.columns

    # Undersampling
    n_p_us = int(np.ceil(n_p/us_factor))

    # Work wih matrices (better)
    processed_mat = np.zeros((n_p_us, len(feat_list)))

    for i in range(n_p_us):
        # Average the measurements at groups of 'us_factor'
        # final index (+1) for considered group
        end_ind = min(n_p, (i+1)*us_factor)
        index_list = list(range(i*us_factor, end_ind))

        current_subset = df_start.iloc[index_list]

        # Take mean of measurement
        rows_avg = current_subset.mean(axis=0)

        # Replace the specified feature with the variance measured over the current subset
        for feat in takeVar:
            var_curr = current_subset[feat].var(axis=0)
            rows_avg = rows_avg.drop(labels=[feat])
            newname = 'var_'+str(feat)
            # newname = str(feat)
            rows_avg[newname] = var_curr

        # Place the created feature at the end of the data matrix
        feat_list = rows_avg.index
        processed_mat[i, :] = np.copy(rows_avg.values)

    n_p_processed = processed_mat.shape[0]

    # DBSCAN
    if dbscan:
        clustering = sk.DBSCAN(
            eps=dbscan_eps, min_samples=dbscan_M).fit(processed_mat)
        ind_outlier = np.argwhere(clustering.labels_ == -1)[:, 0]
        n_removed = len(ind_outlier)
        print(f"DBSCAN found {n_removed} outliers")
        processed_mat = np.delete(processed_mat, ind_outlier, 0)
        n_p_processed = processed_mat.shape[0]

    df_proc = pd.DataFrame(processed_mat, columns=feat_list)

    # MSV of features:
    if (len(msv_list[0]) > 0):
        for i in range(len(msv_list)):
            curr_list = msv_list[i]
            tmp_sum = np.zeros((n_p_processed,))
            for col in curr_list:
                tmp_sum += (df_proc[col].values)**2

            df_proc = df_proc.drop(columns=curr_list)
            col_name = 'msv_'+str(i)
            df_proc[col_name] = np.sqrt(tmp_sum)

    # May be unnecessary
    # Variance normalization:
    # Normalize feature variance only if it's above the thresh
    if var_norm:
        stdev = df_proc.std(axis=0)
        stdev[stdev < np.sqrt(var_thresh)] = 1
        df_proc = (df_proc/stdev)
    ######

    return df_proc

#
#
#
#
def buildDataSet(filedir, patient, activities, slices, all_sensors, all_sensors_names, sensors_tbr_names, sensors_tba, ID='train'):
    """
    buildDataSet
    ---------------------------------------------------------
    Returns the dataframe containing the features and the 
    column containing the corresponding label.
    ---------------------------------------------------------
    Parameters:
    - activities: array of activity IDs
    - slices: array of slice IDs
    - all_sensors: list of sensors IDs to be read
    - all_sensor_names: list of sensor names to be read
    - sensors_tbr_names: names of sensors to be removed at 
      preprocessing
    - sensors_tba: sensors that need to be averaged at 
      preprocessing (to be removed since no averaging)
    ---------------------------------------------------------
    """
    
    if ID == 'train':
        dbscan = True
        var_norm = True
    else:
        dbscan = False
        var_norm = False


    n_clusters = len(activities)
    n_sensors_tot = len(all_sensors)

    filedir2 = './lab04/' + str(filedir)
    
    created = False
    n_features = n_sensors_tot - len(sensors_tbr_names)
    start_centroids = np.zeros((n_clusters, n_features))

    ## Open one activity at a time to allow for individual preprocessing of all classes
    # This way we can also obtain starting centroids for k-means as the average element 
    # of each class
    for act in activities:
        try:
            x_curr = generateDF(filedir, all_sensors_names, all_sensors, patient, [act], slices)
        except:
            x_curr = generateDF(filedir2, all_sensors_names, all_sensors, patient, [act], slices)

        labels_curr = x_curr.activity

        x_curr = x_curr.drop(columns=['activity'])
        # Preprocess (same parameters as before) - need to pass elements 
        # without class, else the label is modified (processed...)
        x_curr = preprocessor(x_curr, drop_feat=sensors_tbr_names, us_factor=25, dbscan=dbscan,
                            dbscan_eps=0.7, dbscan_M=6, var_norm=var_norm)  # (Nslices*125)x(n_sensors)

        # Centroid i corresponds to class i+1
        start_centroids[act-1, :] = x_curr.mean().values

        # Replace labels
        x_curr['activity'] = labels_curr[0]

        if not created:
            x_tr_df = x_curr.copy()
            created = True
        else:
            x_tr_df = pd.concat([x_tr_df, x_curr])

    X_created = x_tr_df.drop(columns=['activity']).values
    y_created = x_tr_df.activity.values

    return X_created, y_created, start_centroids