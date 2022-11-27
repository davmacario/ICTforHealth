import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def C(x):
    X1 = x[0]
    X2 = x[1]
    return np.sign(-2*np.sign(X1)*np.power(np.abs(X1), 2/3) + 4*(X2**2))


Nf = 2
N_train = 1000
# Since function rand() generates samples in [0,1], need to fit them in [-1, 1]
X_tr = 2*np.random.rand(N_train, Nf) - np.ones((N_train, Nf))
C_tr = np.zeros((N_train,))
for i in range(N_train):
    C_tr[i] = C(X_tr[i, :])

X_tr_b = X_tr[C_tr == +1, :]
X_tr_r = X_tr[C_tr == -1, :]

plt.figure()
plt.plot(X_tr_b[:, 0], X_tr_b[:, 1], '.b', label='C = +1')
plt.plot(X_tr_r[:, 0], X_tr_r[:, 1], '.r', label='C = -1')
plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
plt.title("Training set")
plt.show()

clfX = tree.DecisionTreeClassifier(criterion='entropy')
clfX = clfX.fit(X_tr, C_tr)

#####
N_test = 20000
X_test = 2*np.random.rand(N_test, Nf) - np.ones((N_test, Nf))

C_test = np.zeros((N_test,))
for i in range(N_test):
    C_test[i] = C(X_test[i, :])

C_hat_test = clfX.predict(X_test)

acc_te = accuracy_score(C_test, C_hat_test)

print(f"Accuracy = {acc_te}")

X_te_b = X_test[C_hat_test == +1, :]
X_te_r = X_test[C_hat_test == -1, :]

plt.figure()
plt.plot(X_te_b[:, 0], X_te_b[:, 1], '.b', label=r'\hat{C} = +1')
plt.plot(X_te_r[:, 0], X_te_r[:, 1], '.r', label=r'\hat{C} = -1')
plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
plt.title("Test set")
plt.show()


#
