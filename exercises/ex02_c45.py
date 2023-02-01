import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def C(x):
    """
    Used to get the class label for each point x = (x_1, x_2)
    """
    X1 = x[0]
    X2 = x[1]
    return np.sign(-2*np.sign(X1)*np.power(np.abs(X1), 2/3) + 4*(X2**2))


Nf = 2
N_train = 1000
#N_train = 10000
# Since function rand() generates samples in [0,1], need to fit them in [-1, 1]
X_tr = 2*np.random.rand(N_train, Nf) - np.ones((N_train, Nf))

# Use the following method to find the class of the generated points
# in one line
C_tr = np.apply_along_axis(C, 1, X_tr)

X_tr_b = X_tr[C_tr == +1]
X_tr_r = X_tr[C_tr == -1]

plt.figure()
plt.plot(X_tr_b[:, 0], X_tr_b[:, 1], '.b', label='C = +1')
plt.plot(X_tr_r[:, 0], X_tr_r[:, 1], '.r', label='C = -1')
plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
plt.title("Training set")
try:
    plt.savefig('./img/training_set.png')
except:
    plt.savefig('./exercises/img/training_set.png')
plt.show()

decision_tree_classifier = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree_classifier_fitted = decision_tree_classifier.fit(X_tr, C_tr)

#####
N_test = 20000
X_test = 2*np.random.rand(N_test, Nf) - np.ones((N_test, Nf))

C_test = np.zeros((N_test,))
for i in range(N_test):
    C_test[i] = C(X_test[i, :])

C_hat_test = decision_tree_classifier_fitted.predict(X_test)

acc_te = accuracy_score(C_test, C_hat_test)

print(f"Accuracy = {acc_te}")

X_te_b = X_test[C_hat_test == +1]
X_te_r = X_test[C_hat_test == -1]

plt.figure()
plt.plot(X_te_b[:, 0], X_te_b[:, 1], '.b', label=r'\hat{C} = +1')
plt.plot(X_te_r[:, 0], X_te_r[:, 1], '.r', label=r'\hat{C} = -1')
plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
plt.title("Test set")
try:
    plt.savefig('./img/test_set.png')
except:
    plt.savefig('./exercises/img/test_set.png')
plt.show()

# Print the tree
feat_names = ["X_1", "X_2"]
target_names = ["C=-1", "C=+1"]

fig, axes = plt.subplots(figsize=(10, 10))
tree.plot_tree(decision_tree_classifier_fitted,
               feature_names=feat_names,
               class_names=target_names,
               rounded=True,
               filled=True)
try:
    fig.savefig("img/tree.png")
except:
    fig.savefig("exercises/img/tree.png")
