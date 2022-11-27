"""
First example - solving LLS problem - no classes created
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(315054)

Np = 10      # rows
Nf = 5      # features

A = np.random.randn(Np, Nf)
w = np.random.randn(Nf)
y = A@w

# %% Evaluate w hat
# Apply the linear least squares method
ATA = A.T@A     # Generate A^T * A
ATA_inv = np.linalg.inv(ATA)        # Inverse of A^T * A
ATy = A.T@y     # Generate A^T * y

w_hat = ATA_inv@ATy

# %% Check result
e = y - A@w_hat

errsqnorm = np.linalg.norm(e)**2

print("Error square norm: ", errsqnorm)

# %%


plt.figure()        # Create the figure
plt.plot(w_hat, label='w_hat')    # w_hat will be plotted as a line
plt.plot(w, 'o', label='w')   # w will be plotted with 'o's as markers
plt.xlabel('n')
plt.ylabel('w(n)')
plt.legend()
plt.grid()
plt.title("Comparison between w and w_hat")
plt.show()
