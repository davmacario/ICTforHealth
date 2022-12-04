import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

""" 
- TODO[1]: Conclude draft program - look at all plots/features/activities
- TODO[2]: Decide processing strategy
- TODO[3]: Implement processing strategy (object-oriented if possible)
- TODO[4]: Look for better results than in slide 26
- TODO[5]: Separate processed data in training and test sets and apply sklearn 
- TODO[6]: Find mapping between cluster and activity to check results
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

def generateDF(filedir, colnames, patients, activities, slices):
    # get the data from files for the selected patients
    # and selected activities
    # concatenate all the slices
    # generate a pandas dataframe with an added column: activity
    x=pd.DataFrame()
    for pat in patients:
        for a in activities:
            subdir='a'+f"{a:02d}"+'/p'+str(pat)+'/'
            for s in slices:
                filename=filedir+subdir+'s'+f"{s:02d}"+'.txt'
                x1=pd.read_csv(filename,names=colnames)
                x1['activity']=a*np.ones((x1.shape[0],),dtype=int)
                x=pd.concat([x,x1], axis=0, join='outer', ignore_index=True, 
                            keys=None, levels=None, names=None, verify_integrity=False, 
                            sort=False, copy=True)
    return x

plt.close('all')
cm = plt.get_cmap('gist_rainbow')
line_styles=['solid','dashed','dotted']

filedir1 = './data/'
filedir2 = './lab04/data/'

sensNames=[
        'T_xacc', 'T_yacc', 'T_zacc', 
        'T_xgyro','T_ygyro','T_zgyro',
        'T_xmag', 'T_ymag', 'T_zmag',
        'RA_xacc', 'RA_yacc', 'RA_zacc', 
        'RA_xgyro','RA_ygyro','RA_zgyro',
        'RA_xmag', 'RA_ymag', 'RA_zmag',
        'LA_xacc', 'LA_yacc', 'LA_zacc', 
        'LA_xgyro','LA_ygyro','LA_zgyro',
        'LA_xmag', 'LA_ymag', 'LA_zmag',
        'RL_xacc', 'RL_yacc', 'RL_zacc', 
        'RL_xgyro','RL_ygyro','RL_zgyro',
        'RL_xmag', 'RL_ymag', 'RL_zmag',
        'LL_xacc', 'LL_yacc', 'LL_zacc', 
        'LL_xgyro','LL_ygyro','LL_zgyro',
        'LL_xmag', 'LL_ymag', 'LL_zmag']

actNames=[
    'sitting',  # 1
    'standing', # 2
    'lying on back',# 3
    'lying on right side', # 4
    'ascending stairs' , # 5
    'descending stairs', # 6
    'standing in an elevator still', # 7
    'moving around in an elevator', # 8
    'walking in a parking lot', # 9
    'walking on a treadmill with a speed of 4 km/h in flat', # 10
    'walking on a treadmill with a speed of 4 km/h in 15 deg inclined position', # 11
    'running on a treadmill with a speed of 8 km/h', # 12
    'exercising on a stepper', # 13
    'exercising on a cross trainer', # 14
    'cycling on an exercise bike in horizontal positions', # 15
    'cycling on an exercise bike in vertical positions', # 16
    'rowing', # 17
    'jumping', # 18
    'playing basketball' # 19
    ]

actNamesShort=[
    'sitting',  # 1
    'standing', # 2
    'lying.ba', # 3
    'lying.ri', # 4
    'asc.sta' , # 5
    'desc.sta', # 6
    'stand.elev', # 7
    'mov.elev', # 8
    'walk.park', # 9
    'walk.4.fl', # 10
    'walk.4.15', # 11
    'run.8', # 12
    'exer.step', # 13
    'exer.train', # 14
    'cycl.hor', # 15
    'cycl.ver', # 16
    'rowing', # 17
    'jumping', # 18
    'play.bb' # 19
    ]


#%%##########################################################################################

student_ID = 315054
s = student_ID%8 + 1    ## Used subject

patients = [s]
activities = list(range(1,6))           # List of indexes of activities to plot (TO BE TUNED)
Num_activities = len(activities)        # Number of considered activities
NAc = 19                                # Total number of activities
actNamesSub = [actNamesShort[i-1] for i in activities] # short names of the selected activities

sensors = list(range(9))                # List of sensors (TO BE TUNED)
sensNamesSub = [sensNames[i] for i in sensors] # Names of selected sensors

Nslices = 12                            # Number of slices to plot (TO BE TUNED)
Ntot = 60                               # Total number of slices
slices = list(range(1,Nslices+1))       # First Nslices to plot

fs = 25                                 # Hz, sampling frequency (fixed)
samplesPerSlice = fs*5                  # Samples in each slice (fixed)

#%%##########################################################################################
# Plot centroids and stand. dev. of sensor values
# All activities considered, but only 9 features

print('Number of used sensors: ',len(sensors))

centroids = np.zeros((NAc,len(sensors)))        # Centroids for all the activities
stdpoints = np.zeros((NAc,len(sensors)))        # Variance in cluster for each sensor

plt.figure(figsize=(12,6))

for i in range(1, NAc + 1):
    activities = [i]
    try:
        x = generateDF(filedir1,sensNamesSub,patients,activities,slices)
    except:
        x = generateDF(filedir2,sensNamesSub,patients,activities,slices)
    x = x.drop(columns=['activity'])
     
    centroids[i-1,:]=x.mean().values

    plt.subplot(1,2,1)
    lines = plt.plot(centroids[i-1,:],label=actNamesShort[i-1])
    lines[0].set_color(cm(i//3*3/NAc))
    lines[0].set_linestyle(line_styles[i%3])

    stdpoints[i-1] = np.sqrt(x.var().values)
    
    plt.subplot(1,2,2)
    lines = plt.plot(stdpoints[i-1,:],label=actNamesShort[i-1])
    lines[0].set_color(cm(i//3*3/NAc))
    lines[0].set_linestyle(line_styles[i%3])

plt.subplot(1,2,1)
plt.legend(loc='upper right')
plt.grid()
plt.title('Centroids using '+str(len(sensors))+' sensors')
plt.xticks(np.arange(x.shape[1]),list(x.columns),rotation=90)

plt.subplot(1,2,2)
plt.legend(loc='upper right')
plt.grid()
plt.title('Standard deviation using '+str(len(sensors))+' sensors')
plt.xticks(np.arange(x.shape[1]),list(x.columns),rotation=90)

plt.tight_layout()
plt.show()

#%%##########################################################################################\
# Inter-centroid distance
d = np.zeros((NAc, NAc))        # Element i, j is squared norm of dist between centroid i and j

for i in range(NAc):
    for j in range(i+1, NAc):       # Optimization - diagonal elements are 0 and matrix is symmetric
        d[i, j] = np.linalg.norm(centroids[i] - centroids[j])
        d[j, i] = d[i, j]

plt.matshow(d)
plt.colorbar()
plt.xticks(np.arange(NAc),actNamesShort,rotation=90)
plt.yticks(np.arange(NAc),actNamesShort)
plt.title('Between-centroids distance')
plt.show()