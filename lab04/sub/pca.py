import numpy as np
import pandas as pd


class PCA:
    """
    PCA
    ----------------------------------------------------
    Class used to perform Principal Component Analysis
    ----------------------------------------------------
    """


    def __init__(self, X):
        
        if isinstance(X, np.ndarray):
            self._type = 'numpy.ndarray'
            self.X = X
        elif self._type == isinstance(X, pd.DataFrame): 
            self._type = 'pandas.DataFrame'
            self.X = X.values
        else:
            raise TypeError(f"Accepted types are:\n'numpy.ndarray'\n'pandas.DataFrame', not {self._type}!")
        
        self.Np, self.Nf = X.shape
        
        # Need to remove the mean for PCA
        self._X_mean = np.mean(self.X, axis=0)
        self.X_noMean = self.X - self._X_mean

        # Sigma: sample covariance matrix
        tmp_sum = np.zeros((self.Nf, self.Nf))
        for i in range(self.Np):
            x_i = self.X[i, :].reshape((self.Nf, 1))
            tmp_sum += x_i@x_i.T
        
        self.Sigma = tmp_sum/self.Np

        eval, evect = np.linalg.eig(self.Sigma)
        # Sort eigenvalues
        ind_sort = np.argsort(-1*eval)
        
        # self.eval and self.evect are the sorted version!
        self.eval = eval[ind_sort]
        # self.evect contains as columns the orthonormal vectors ordered by 
        self.evect = evect[:, ind_sort]


    def reduce_dim(self, X_te, n_dim):
        # Check same dim. as self.X
        if X_te.shape[1] != self.Nf:
            raise ValueError("The number of features does not match!")
        
        # Remove mean
        X_te_noMean = X_te - self._X_mean
        
        W = self.get_W(n_dim)
        # Multiply by the correct slice of self.evect
        
        X_te_PCA = np.zeros((X_te_noMean.shape[0], n_dim))

        for i in range(X_te_noMean.shape[0]):
            x_i = X_te_noMean[i, :].reshape((self.Nf, 1))
            z_i = W.T@x_i
            X_te_PCA[i, :] = np.copy(z_i.T)
        
        return X_te_PCA


    def get_W(self, n_dim):
        """
        Return the transformation matrix used for reducing the features
        from Nf (original dataset) to a specified n_dim
        """
        return self.evect[:, :n_dim]

if __name__ == "__main__":
    X = np.random.randn(40, 10)
    pca = PCA(X)
    X_pca = pca.reduce_dim(X, 5)

    print(X)
    print(X_pca)