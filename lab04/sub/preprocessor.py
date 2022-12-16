import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk

from scipy import signal



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

cm = plt.get_cmap('gist_rainbow')
line_styles = ['solid', 'dashed', 'dotted']



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

def preprocessor(df, us_factor=1, takeVar=[], lp=False, fs=25, lp_freq=10, \
    dbscan=False, dbscan_eps=1, dbscan_M=5, msv_list=[[]], var_norm=False, var_thresh=1):
    """
    preprocessor
    --------------------------------------------------------------
    Apply a preprocessing pipeline to an input dataframe.

    Possible preprocessing strategies:
    - Undersampling
        - When downsampling, replace feature with variance
    - Mean squared value of x, y, z of each measure
    - Normalize variance of selected features
    --------------------------------------------------------------
    Input parameters:
    - df: input dataframe
    - us_factor: undersampling factor
    - takeVar: flag for substituting average feature with 
      variance (also if variance = 0)
    - lp: flag for using a low-pass filter (Butterworth 2nd order)
    - fs: sampling frequency of the signal
    - lp_freq: cutoff frequency of the lpf
    - dbscan: flag for performing dbscan
    - dbscan_eps: hypersphere radius of DBSCAN
    - dbscan_M: number of neighbors dbscan
    - msv_list = list of lists including the features to be 
      replaced with their mean square value (after undersampling)
    - var_norm: flag for performing variance normalization
    - var_thresh: value of the variance above which normalization is 
      performed
    --------------------------------------------------------------
    """

    df_start = df.copy()
    n_p, n_f = df.shape

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
            if us_factor > 1:
                var_curr = current_subset[feat].var(axis=0)
                rows_avg = rows_avg.drop(labels=[feat])
                newname = 'var_'+str(feat)
                # newname = str(feat)
                rows_avg[newname] = var_curr

        # Place the created feature at the end of the data matrix
        feat_list = rows_avg.index
        processed_mat[i, :] = np.copy(rows_avg.values)

    n_p_processed = processed_mat.shape[0]

    # LPF:
    if lp:
        for i in range(n_p_processed):
            # Notice - the sampling frequency is like if it was reduced when averaging (undersampling)
            processed_mat[i, :] = np.copy(custom_filter(processed_mat[i, :], fs/us_factor, lp_freq/us_factor))

    # DBSCAN
    if dbscan:
        clustering = sk.DBSCAN(
            eps=dbscan_eps, min_samples=dbscan_M).fit(processed_mat)
        ind_outlier = np.argwhere(clustering.labels_ == -1)[:, 0]
        n_removed = len(ind_outlier)
        # print(f"DBSCAN found {n_removed} outliers")
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


def denoiser(x, cutoff):
    """
    denoiser
    ----------------------------------------------------
    fft-based simple low-pass filter
    ----------------------------------------------------
    """
    n_elem = len(x)
    
    x_fft = np.fft.fft(x, n_elem)

    PSD = x_fft*np.conj(x_fft)/n_elem

    _mask = PSD > cutoff
    x_fft_filt = x_fft * _mask

    x_filt = np.fft.ifft(x_fft_filt).real

    return x_filt

def custom_filter(x, fs, cutoff):
    """
    custom_filter
    ----------------------------------------------------
    applies the Butterworth filter of order 2
    ----------------------------------------------------
    - x: signal
    - fs: sampling frequency
    - cutoff: cutoff frequency (or tuple if bp)
    """

    f_ny = fs/2

    b, a = signal.butter(2, cutoff, btype='lowpass', fs=fs)

    x_filt = np.array(signal.filtfilt(b, a, x))

    return x_filt


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
#
#
#
#
def buildDataSet(filedir, patient, activities, slices, all_sensors, all_sensors_names, sensors_var, ID='train', plots=True):
    # TODO: add plots (flags) + save them
    """
    buildDataSet
    ---------------------------------------------------------
    Returns the dataframe containing the features and the 
    column containing the corresponding label.
    ---------------------------------------------------------
    Parameters:
    - filedir: difectory of the input files
    - patient: considered patient
    - slices: considered slices (list of int)
    - activities: array of activity IDs
    - slices: array of slice IDs
    - all_sensors: list of sensors IDs to be read
    - all_sensor_names: list of sensor names to be read
    - sensors_var: sensors whose measurements will be 
      substituted with their variance
    - ID: determines whether this Data Set is a training set 
      of test set to prevent forbidden operations on the test
      set
    - plots: flag for plotting figures
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
    
    n_features = n_sensors_tot

    start_centroids = np.zeros((n_clusters, n_features))
    stdpoints = np.zeros((n_clusters, n_features))

    if plots:
        plt.figure(figsize=(12, 6))

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
        x_curr = preprocessor(x_curr, us_factor=50, lp=False, fs=25, lp_freq=12, dbscan=dbscan,
                            dbscan_eps=1.2, dbscan_M=6, var_norm=var_norm)  # (Nslices*125)x(n_sensors)

        # Centroid i corresponds to class i+1
        start_centroids[act-1, :] = x_curr.mean().values

        # Standard deviation
        stdpoints[act-1] = np.sqrt(x_curr.var().values)

        # Replace labels
        x_curr['activity'] = labels_curr[0]

        if not created:
            x_tr_df = x_curr.copy()
            created = True
        else:
            x_tr_df = pd.concat([x_tr_df, x_curr])

        if plots:
            plt.subplot(1, 2, 1)
            lines = plt.plot(start_centroids[act-1, :], label=actNamesShort[act-1])
            lines[0].set_color(cm(act//3*3/n_clusters))
            lines[0].set_linestyle(line_styles[act % 3])

            plt.subplot(1, 2, 2)
            lines = plt.plot(stdpoints[act-1, :], label=actNamesShort[act-1])
            lines[0].set_color(cm(act//3*3/n_clusters))
            lines[0].set_linestyle(line_styles[act % 3])
    
    if plots:
        plt.subplot(1, 2, 1)
        plt.legend(loc='upper right')
        plt.grid()
        plt.title('Centroids using '+str(n_sensors_tot)+' sensors')
        plt.xticks(np.arange(x_curr.shape[1]), list(x_curr.columns), rotation=90)

        plt.subplot(1, 2, 2)
        plt.legend(loc='upper right')
        plt.grid()
        plt.title('Standard deviation using '+str(n_sensors_tot)+' sensors')
        plt.xticks(np.arange(x_curr.shape[1]), list(x_curr.columns), rotation=90)

        plt.tight_layout()
        plt.show()

    X_created = x_tr_df.drop(columns=['activity']).values
    y_created = x_tr_df.activity.values

    return X_created, y_created, start_centroids, stdpoints