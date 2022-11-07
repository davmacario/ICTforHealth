import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

np.random.seed(315054)

Np = 200        # Number of points in the Gaussian random process
Nm = 21         # FIR filter memory
Nprev = 2*Nm    # Time axis limits
T = 10

th = np.arange(-Nprev, Nprev)   # Considered time axis - 4* (filter memory)

h = np.exp(-(th/T)**2)      # Gaussian Impulse Response
h = h/np.linalg.norm(h)
x = np.random.randn(Np,)    # Imput WGN
y = np.convolve(x, h, mode='same')      # Output filtered Gaussian process
t = np.arange(len(y))       # Time axis of the output signal (0 to length of y)

# Plot the impulse response of the filter and the realization of the random process
plt.figure()
plt.plot(th, h)
plt.grid()
plt.xlabel('t (s)')
plt.ylabel('h(t)')
plt.show()

plt.show()
plt.plot(t, y)
plt.grid()
plt.xlabel('t (s)')
plt.ylabel('y(t)')
plt.title('Realization of the filtered Gaussian random process')
plt.show()

# Autocorrelation
autocorr = np.exp(-(th/T)**2/2)

plt.figure()
plt.plot(th, autocorr)
plt.xlabel(r'$\tau (s)$')
plt.ylabel(r'$R_Y(\tau)$')
plt.title('Autocorrelation function')
plt.grid()
plt.show()

# Select 10 random points in time (t) of (y)
M_sampled = 10
# Create a vector of size (M_sampled) taken from t, without replacement
t_sampled = np.random.choice(t, (M_sampled,), replace=False)
y_sampled = y[t_sampled]

# Select the sample t_* randomly among the ones which were not selected
t_rem = list(set(t) - set(t_sampled))
t_star = np.random.choice(t_rem, (1,), replace=False)
y_true = y[t_star]

# Create covariance matrix
# First, create matrix containing time differences
# Subtract the vector transpose from itself, but after reshape
t_new = t[:, np.newaxis]        # Make the array t 2D
delta_t_matr = t_new - t_new.T

print(delta_t_matr)

# Generating the Full Covariance matrix
R = np.exp(-(delta_t_matr/T)**2/2)

plt.matshow(R)
plt.colorbar()
plt.title('Theoretical Covariance matrix')
plt.show()

# Generalte the Covariance matrix starting from the extracted samples
t_sampled_resh = t_sampled[:, np.newaxis]
delta_t_matr_N = t_sampled_resh - t_sampled_resh.T

R_N = np.exp(-(delta_t_matr_N/T)**2/2)
plt.matshow(R_N)
plt.colorbar()
plt.title('Covariance matrix - over the training samples')
plt.show()

# Find the approx. value of y(t_*) by using Gaussian Process regression

# Find the approx.value of y(t_*) by using linear regression (with LLS)

# Final plot
plt.figure()
plt.plot(t, y, 'b-', label='Realization')
plt.plot(t_sampled, y_sampled, 'bo', label='Sampled values')
plt.plot(t_star, y_true, 'ro', label='True value')
plt.xlabel('t (s)')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()
