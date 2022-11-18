import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as sk
plt.rcParams["font.family"] = "Times New Roman"
#%%
plt.close('all')
xx=pd.read_csv("covid_serological_results.csv")
swab=xx.COVID_swab_res.values# results from swab: 0= no illness, 1 = unclear, 2=illness
swab[swab>=1]=1# non reliable values are considered positive
data_descr=False
#%% data analysis

if data_descr:
    xx_pos=xx[xx.COVID_swab_res==1]
    xx_neg=xx[xx.COVID_swab_res==0]
    xx_pos=xx_pos.drop(columns=['COVID_swab_res'])
    xx_neg=xx_neg.drop(columns=['COVID_swab_res'])
    xx_pos.hist(bins=50)
    xx_neg.hist(bins=50)
    pd.plotting.scatter_matrix(xx_pos)
    pd.plotting.scatter_matrix(xx_neg)
    xx_norm=(xx-xx.mean())/xx.std()
    c=xx_norm.corr()
    plt.figure()
    plt.matshow(np.abs(c.values),fignum=0)# absolute value of corr.coeffs
    plt.xticks(np.arange(3), xx.columns, rotation=90)
    plt.yticks(np.arange(3), xx.columns, rotation=0)    
    plt.colorbar()
    plt.title('Correlation coefficients of the original dataset')
    plt.tight_layout()

