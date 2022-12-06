import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk



def preprocessor(df, drop_feat = [], us_factor=1, meanSqXYZ=False, dbscan=False, dbscan_eps = 1, dbscan_M = 5):
    """
    preprocessor
    ---
    Apply a preprocessing pipeline to an input dataframe
    ---
    Possible preprocessing strategies:
    - Drop features
    - Undersampling
    - Mean squared value of x, y, z of each measure
    """

    df_start = df.copy()
    n_p, n_f = df.shape

    # Drop features
    df_start = df_start.drop(columns=drop_feat)

    feat_list = df_start.columns

    # Undersampling
    n_p_us = int(np.ceil(n_p/us_factor))

    processed_mat = np.zeros((n_p_us, len(feat_list)))  # Work wih matrices (better)
    
    for i in range(n_p_us):
        # Average the measurements at groups of 'us_factor'
        end_ind = min(n_p, (i+1)*us_factor)    # final index (+1) for considered group
        index_list = list(range(i*us_factor, end_ind))
        # Take mean of measurement
        rows_avg = df_start.iloc[index_list].mean().values
        # Place the created feature at the end of the data matrix
        processed_mat[i, :] = np.copy(rows_avg)

    n_p_processed = processed_mat.shape[0]

    ## DBSCAN
    if dbscan:
        clustering = sk.DBSCAN(eps=dbscan_eps, min_samples=dbscan_M).fit(processed_mat)
        ind_outlier = np.argwhere(clustering.labels_ == -1)[:, 0]
        n_removed = len(ind_outlier)
        print(f"DBSCAN found {n_removed} outliers")
        processed_mat = np.delete(processed_mat, ind_outlier, 0)
        n_p_processed = processed_mat.shape[0]



    df_proc = pd.DataFrame(processed_mat, columns=feat_list)

    return df_proc





        