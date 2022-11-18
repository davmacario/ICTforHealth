import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk

plt.rcParams["font.family"] = "Times New Roman"
#%%
plt.close('all')

inputFile = 'data/covid_serological_results.csv'
inputFile_2 = 'lab03/data/covid_serological_results.csv'

try:
    xx = pd.read_csv(inputFile)
except:
    xx = pd.read_csv(inputFile_2)

features = xx.columns

# 0 = no illness; 1 = unclear; 2 = ill
swab=xx.COVID_swab_res.values
swab[swab>=1]=1# non reliable values are considered positive
data_descr=False

## Plot test 2 vs. test 1 - see outliers


## Carry out DBSCAN, setting suitable parameters (eps, M)


## Drop the outliers


##