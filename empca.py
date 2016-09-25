"""Expectation-Maximization for Principal Component Analysis"""
# Author: Artsem Zhuk


from sklearn.decomposition import PCA
import numpy as np
#from scipy import linalg
from numpy.linalg import inv
from numpy import dot, transpose, trace
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from utils import gram_schmidt, normalize


class EMPCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_components, copy=False,
                 n_iter=100):
        self.n_components = n_components
        self.copy = copy
        self.n_iter = n_iter

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: (n_samples, n_features)

        Returns
        -------
        self: object
        """
        self._fit(X)
        return self

    def _fit(self, X):
        X = check_array(X, dtype=[np.float64], ensure_2d=True, copy=self.copy)

        n_samples, n_features = X.shape
        n_components = self.n_components

        self.mean_ = normalize(X)

        sigma = 1
        W = np.zeros([n_features, n_components], dtype=np.float64)
        for i in range(n_features):
            for j in range(n_components):
                W[i][j] = np.random.uniform()

        x_sum = sum([dot(X[i], transpose(X[i])) for i in range(n_samples)])

        for _ in range(self.n_iter):
            # e-step
            Wt = transpose(W)
            M = inv(dot(Wt, W) + sigma * np.eye(n_components))

            Sigma = sigma * M

            mu = dot(X, W)
            mu = dot(mu, transpose(M))

            # m-step
            L = dot(transpose(X), mu)
            R = dot(transpose(mu), mu) + n_samples * Sigma  #  sum Ett

            W_new = dot(L, inv(R))
            W_new_t = transpose(W_new)


            C = trace(dot(dot(W_new_t, W_new), R))
            B = sum([dot(dot(mu[i], W_new_t), transpose(X[i])) for i in range(n_samples)])
            A = x_sum
            #   print(A, B, C)
            sigma_new = (A - 2 * B + C) / n_samples / n_features

            W = gram_schmidt(W_new, True)
            sigma = sigma_new

            #print("W=\n", W)
            #print("sigma=", sigma)
            #print()

        self.components_ = gram_schmidt(transpose(W))
        self.sigma_ = sigma

    def transform(self, X, y=None):
        X = check_array(X, dtype=[np.float64], ensure_2d=True, copy=self.copy)
        X -= self.mean_

        return dot(X, transpose(self.components_))

    def fit_transform(self, X, y=None):
        self._fit(X)
        normalize(X)
        X = dot(X, transpose(self.components_))
        return X


if __name__ == '__main__':
    X = np.array([[1, 2], [2, 5], [5, 7]], dtype=np.float64)
    #normalize(X)
    #print(X)

    empca = EMPCA(n_components=2, n_iter=3000)
    empca.fit(X)
    print(empca.components_)
    #print(empca.transform(X))

    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.components_)

    exit(0)

    pca = PCA(n_components=1, copy=False)
    print(pca.fit(X))

    Z = np.array([[0, 1], [1, 0], [5, 5]])

    print(pca.transform(Z))

    print(pca.mean_)
    print(pca.components_)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)