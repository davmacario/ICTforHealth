import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk


def preprocessor(df, drop_feat=[], us_factor=1, takeVar=[], dbscan=False, dbscan_eps=1, dbscan_M=5, msv_list=[[]], var_norm=False, var_thresh=1):
    """
    preprocessor
    ---
    Apply a preprocessing pipeline to an input dataframe.

    Possible preprocessing strategies:
    - Drop features
    - Undersampling
        - When downsampling, replace feature with variance
    - Mean squared value of x, y, z of each measure
    - Normalize variance of selected features
    ---
    Input parameters:
    - df: input dataframe
    - drop_feat: list of features of the dataframe to be dropped
    - us_factor: undersampling factor
    - takeVar: flag for substituting average feature with variance (WHAT IF VAR = 0?)
    - dbscan: flag for performing dbscan
    - dbscan_eps: hypersphere radius of DBSCAN
    - dbscan_M: number of neighbors dbscan
    - msv_list = list of lists including the features to be replaced with their mean 
      square value (after undersampling)

    - var_norm: flag for performing variance normalization
    - var_eps: value of the variance above which normalization is performed
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
