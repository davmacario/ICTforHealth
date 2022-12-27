import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk
from scipy import signal

#
#
#
#
#
#
#
# Global variables used in the programs
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
#
#
#
#
#
#
# Generic functions used in the program
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
#
# Preprocessing pipeline - implemented as a 
# class to allow for evaluating the filter 
# parameters just once
class Preprocessor:
    def __init__(self, fs, filt_type, cutoff, us_factor=25):
        """
        Preprocessor
        --------------------------------------------------------------
        This class is used to initialize and apply on any dataset a 
        preprocessing pipeline based on time-series filtering and 
        undersampling (with averaging).
        --------------------------------------------------------------
        Input parameters:
        - fs: sampling frequency for the discrete signals
        - filt_type: type of filter to be applied. Supported values 
          are:
            - `lowpass`
            - `highpass`
            - `bandpass`
            - `bandstop`
            - `none` -> no filter will be applied
        - cutoff: cutoff frequency for the filter. If the filter is 
          either bandpass or bandstop, it must be a length-2 sequence
        - us_factor: undersampling factor, i.e., number of samples to 
          be averaged
        --------------------------------------------------------------
        """
        self.us = us_factor
        self.fs = fs
        self.cutoff = cutoff
        self.filt_type = filt_type
        if filt_type == 'none':
            self.filt = False
        else:
            self.filt = True
            self._b, self._a = signal.butter(2, cutoff, btype=filt_type, fs=fs)
        

    # Complete preprocessing pipeline
    def transform(self, df):
        """
        transform
        --------------------------------------------------------------
        Apply a preprocessing pipeline to an input dataframe.
        Parameter:
        - df: input dataframe
        --------------------------------------------------------------
        """

        df_start = df.copy()
        n_p, n_f = df.shape
        feat_list = df_start.columns

        start_values = df_start.values

        ############## Filter ################################################################
        if self.filt:
            for i in range(n_f):
                # The signals (time series) are on the columns - 1 signal per sensor
                start_values[:, i] = np.array(signal.filtfilt(self._b, self._a, start_values[:, i]))

            df_start = pd.DataFrame(start_values, columns=feat_list)

        ############## Undersampling
        n_p_us = int(np.ceil(n_p/self.us))

        # Work wih matrices (better)
        processed_mat = np.zeros((n_p_us, len(feat_list)))

        for i in range(n_p_us):
            # Average the measurements at groups of 'us_factor'
            # final index (+1) for considered group
            end_ind = min(n_p, (i+1)*self.us)
            index_list = list(range(i*self.us, end_ind))

            current_subset = df_start.iloc[index_list]

            # Take mean of measurement
            rows_avg = current_subset.mean(axis=0)

            # Place the created feature at the end of the data matrix
            feat_list = rows_avg.index
            processed_mat[i, :] = np.copy(rows_avg.values)

        df_proc = pd.DataFrame(processed_mat, columns=feat_list)
        ######

        return df_proc
#
#
#
#
#
#
#
#
# Function for building the dataset
def buildDataSet(filedir, patient, activities, slices, 
                all_sensors, preprocessor_obj=None, plots=True):

    """
    buildDataSet
    ---------------------------------------------------------
    Returns the dataframe containing the features and the 
    column containing the corresponding label.
    ---------------------------------------------------------
    Parameters:
    - filedir: difectory of the input files
    - patient: considered patient
    - activities: array of activity IDs
    - slices: array of slice IDs
    - all_sensors: list of sensors IDs to be read
    - preprocessor_obj: class Preprocessor object. If not 
      specified a default one will be used
    - plots: flag for plotting figures
    ---------------------------------------------------------
    """
    filedir2 = './lab04/' + str(filedir)

    all_sensors_names = [sensNames[i] for i in all_sensors]
    n_clusters = len(activities)
    n_sensors_tot = len(all_sensors)    
    n_features = n_sensors_tot

    start_centroids = np.zeros((n_clusters, n_features))
    stdpoints = np.zeros((n_clusters, n_features))

    if preprocessor_obj is None:
        preprocessor_obj = Preprocessor(fs=25, filt_type='bandstop', cutoff=[0.01, 12], us_factor=1)

    if plots:
        plt.figure(figsize=(12, 6))

    created = False

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
        x_curr = preprocessor_obj.transform(x_curr)

        # Centroid i corresponds to class i+1
        start_centroids[act-1, :] = x_curr.mean().values

        # Standard deviation
        stdpoints[act-1] = np.sqrt(x_curr.var().values)

        # Replace labels
        x_final = x_curr.copy()
        x_final['activity'] = labels_curr[0]

        if not created:
            x_tr_df = x_final.copy()
            created = True
        else:
            x_tr_df = pd.concat([x_tr_df, x_final])

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