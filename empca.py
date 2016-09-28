"""Expectation-Maximization for Principal Component Analysis"""
# Author: Artsem Zhuk


from sklearn.decomposition import PCA
import numpy as np
#from scipy import linalg
from numpy.linalg import inv
from numpy import dot, transpose, trace
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from utils import gram_schmidt, normalize, project_many, random_model
from copy import copy
import sys

class EMPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, n_iter=100):
        """
        :param n_components: number of latent varbles; dimensionality of subspace to project on
        :param n_iter: number of iterations
        """
        self.n_components = n_components
        self.n_iter = n_iter


    def make_iteration_(self):
        X = self.X_
        n_samples, n_features = X.shape
        n_components = self.n_components
        x_sum = sum([dot(X[i], transpose(X[i])) for i in range(n_samples)])
        W = self.components_.T
        sigma = self.sigma_

        # e-step
        Wt = W.T
        M = inv(dot(Wt, W) + sigma * np.eye(n_components))

        Sigma = sigma * M

        mu = dot(X, W)
        mu = dot(mu, M.T)

        # m-step
        L = dot(transpose(X), mu)
        R = dot(transpose(mu), mu) + n_samples * Sigma  # sum Ett

        W_new = dot(L, inv(R))
        W_new_t = transpose(W_new)

        C = trace(dot(dot(W_new_t, W_new), R))
        B = sum([dot(dot(mu[i], W_new_t), transpose(X[i])) for i in range(n_samples)])
        A = x_sum
        sigma_new = (A - 2 * B + C) / n_samples / n_features

        # W = gram_schmidt(W_new, tr=True)
        # sigma = sigma_new

        self.sigma_ = sigma_new
        self.components_ = gram_schmidt(W_new.T)


    def init_(self, X):
        X = check_array(X, dtype=[np.float64], ensure_2d=True, copy=True)

        self.mean_ = normalize(X)
        self.X_ = X

        self.sigma_ = 1

        n_samples, n_features = X.shape
        n_components = self.n_components

        self.components_ = np.empty([n_components, n_features], dtype=np.float64)
        for i in range(n_components):
            for j in range(n_features):
                self.components_[i][j] = np.random.uniform(-1, 1)

    def calc_ratio_(self):
        X = self.X_
        W = self.components_
        Wl = W.dot(X.T).dot(X)
        lambdas = []

        for i in range(self.n_components):
            l = np.linalg.norm(Wl[i])
            lambdas.append(l)

        x_sum = trace(X.dot(X.T))
        self.lambdas_ = lambdas
        self.explained_variance_ratio_ = lambdas / x_sum

    def fit(self, X, y=None):
        """
        Fit the model with X
        :param X: array-like, n_samples x n_features
        :param y: redundant
        :return: self
        """

        self.init_(X)

        for _ in range(self.n_iter):
            self.make_iteration_()

        self.calc_ratio_()


    def transform(self, X, y=None):
        """
        Projects X onto found subspace
        :param X: vectors to project
        :param y: redundant
        :return:
        """
        X = check_array(X, dtype=[np.float64], ensure_2d=True, copy=True)
        X -= self.mean_
        return X.dot(self.components_.T)


    def inverse_transform(self, Y):
        return Y.dot(self.components_) + self.mean_


    def mean_square_error(self, X):
        Y = X - X.dot(self.components_.T).dot(self.components_)
        t = trace(Y.T.dot(Y))
        t /= X.shape[0]
        return np.sqrt(t)


    def fit_transform(self, X, y=None):
        Xc = copy(X)
        self.fit(X)
        return self.transform(X, y)


if __name__ == '__main__':
    X, W, T = random_model(300, 100, 30) #np.array([[111, 12], [123, 4423], [125, 61]], dtype=np.float64)
    #normalize(X)
    #print(X)

    empca = EMPCA(n_components=5, n_iter=300)
    empca.fit(X)
    print("transformed\n", empca.transform(X))
    #print("components\n", empca.components_)
    print("ratio:\n", empca.explained_variance_ratio_)
    #print(empca.transform(X))
    print('mean_square_error=', empca.mean_square_error(X))

    pca = PCA(n_components=5)
    pca.fit(X)
    #print("pca_components", pca.components_)
    #print("pca_transfomed", pca.transform(X))
    print("pca_ratio", pca.explained_variance_ratio_)

    exit(0)

    pca = PCA(n_components=1)
    print(pca.fit(X))

    Z = np.array([[0, 1], [1, 0], [5, 5]])

    print(pca.transform(Z))

    print(pca.mean_)
    print(pca.components_)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)