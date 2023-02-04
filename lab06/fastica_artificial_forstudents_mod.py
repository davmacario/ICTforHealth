# FastICA on an artificial example

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from matplotlib.pyplot import cm

#np.random.seed(30)  # set the seed to reproduce the experiment

plt.close('all')
#%% Generate the signals
N = 10000 # number of samples
N_signals=4
time = np.linspace(0, 10, N) # generate the time axis (seconds)
time.resize(1,N)
freq1 = 0.5 # main frequency (Hz)
freq2 = freq1*np.sqrt(2)
freq3 = freq1*np.sqrt(5)
x1 = 2*np.sin(2*np.pi*freq1* time-np.pi/4)  # Signal 1: sinusoidal signal
x2 = np.sign(np.sin(2*np.pi*freq2* time-np.pi/5))  # Signal 2: square wave
x3 = 4*signal.sawtooth(2*np.pi*freq3* time)  # Signal 3: saw tooth signal
x4=np.cumsum(x2)
x4=(x4-x4.mean())/x4.std()
x4.resize(1,N) # signal 4: triangular signal

X = np.concatenate((x1,x2,x3,x4),axis=0) # shape: 4 rows N columns
color = iter(cm.Set1(np.linspace(0, 1,N_signals)))
plt.figure()
for n in range(N_signals):
    c = next(color)
    plt.subplot(N_signals,1,1+n)
    plt.plot(X[n,:],'--',color=c,label='original signal '+str(n+1))
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()

#%% Generate the observed data
A = np.random.rand(N_signals,N_signals)  # true weight matrix (random), NxN
Y = np.dot(A,X)  # observed data, shape: 4 rows N columns
color = iter(cm.Set1(np.linspace(0, 1,N_signals)))
plt.figure()
for n in range(N_signals):
    c = next(color)
    plt.subplot(N_signals,1,1+n)
    plt.plot(Y[n,:],'--',color=c,label='mixed signal '+str(n+1))
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()

#%% reshape
X=X.T #shape: N rows, 4 columns 
Y=Y.T #shape: N rows, 4 columns 
#%% look at the pdfs
plt.figure(figsize=(5,8))
for n in range(N_signals):
    plt.subplot(N_signals,1,1+n)
    plt.hist(X[:,n],bins=100,density=True,label='Comp. signal '+str(n))
    plt.legend()
# plt.xlabel('sample values')
# plt.ylabel('pdf')
plt.tight_layout()
plt.figure(figsize=(5,8))
for n in range(N_signals):
    plt.subplot(N_signals,1,1+n)
    plt.hist(Y[:,n],bins=100,density=True,label='Observ. signal '+str(n))
    plt.legend()
plt.tight_layout()
plt.show()
#%% Use FastICA
ica_mod = FastICA(algorithm='deflation', whiten='unit-variance')
X_hat_ICA= ica_mod.fit_transform(Y)

color = iter(cm.Set1(np.linspace(0, 1,N_signals)))
plt.figure()
for n in range(N_signals):
    c = next(color)
    plt.subplot(N_signals,1,1+n)
    plt.plot(X_hat_ICA[:,n]/np.std(X_hat_ICA[:,n]),'-',color=c,label='PCA component '+str(n+1)) 
    plt.plot(X[:,n]/np.std(X[:,n]),'--',color=c,label='signal '+str(n+1))
    plt.grid()
    plt.legend()
plt.title('ICA results')
plt.tight_layout()
try:
    plt.savefig('./img/ICA_res.png')
except:
    plt.savefig('./lab06/img/ICA_res.png')
plt.show()

# Could evaluate the correlation betwen the obtained signals and determine
# the mapping between the found ones and the original ones (permutations
# of the signals order does not change the result!)

#%% Use PCA
pca = PCA(n_components=N_signals)
XhatPCA = pca.fit_transform(Y)  # Reconstruct signals based on orthogonal components
#%% Plot the results
color = iter(cm.Set1(np.linspace(0, 1,N_signals)))
plt.figure()
for n in range(N_signals):
    c = next(color)
    plt.subplot(N_signals,1,1+n)
    plt.plot(XhatPCA[:,n]/np.std(XhatPCA[:,n]),'-',color=c,label='PCA component '+str(n+1)) 
    plt.plot(X[:,n]/np.std(X[:,n]),'--',color=c,label='signal '+str(n+1))
    plt.grid()
    plt.legend()
plt.title('PCA results')
plt.tight_layout()
try:
    plt.savefig('./img/PCA_res.png')
except:
    plt.savefig('./lab06/img/PCA_res.png')
plt.show()
