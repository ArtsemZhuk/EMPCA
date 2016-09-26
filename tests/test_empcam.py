from empca_missing import EMPCAM
from sklearn.decomposition import PCA
import numpy as np
from numpy import dot, transpose
from numpy.linalg import norm
from utils import random_matrix, normalize, random_model, gram_schmidt

import numpy.testing as nt


def show(N, D, d):
    X, W, T = random_model(N, D, d, missing_p=0.1)
    print(X)
    empca = EMPCAM(n_components=d, n_iter=300)
    empca.fit(X)
    print(empca.components_)
    print(gram_schmidt(W))


def show_missing_filling(N):
    W = np.array([[10, 10]]).T

    X, W, T = random_model(N, 2, 1, W=W, missing_p=0.1)

    print(X[:5])
    empca = EMPCAM(n_components=1, n_iter=300)
    empca.fit(X)
    print(empca.components_)
    print('sigma=', empca.sigma_)
    print(empca.fill_missing(X[:5]))


if __name__ == '__main__':
    #show(100, 5, 2)
    show_missing_filling(100)
