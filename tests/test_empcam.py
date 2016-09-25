from empca import EMPCA, normalize
from sklearn.decomposition import PCA
import numpy as np
from numpy import dot, transpose
from numpy.linalg import norm
from utils import random_matrix


import numpy.testing as nt


def assert_almost_parallel(u, v):
    eps = 1e-1
    u /= norm(u)
    v /= norm(v)
    return nt.assert_almost_equal(norm(u - v), 0, decimal=1)


def test_eigen_values(X, n=None):
    empca = EMPCA(n_components=n, n_iter=2000)
    #empca = PCA(n_components=n)
    empca.fit(X)
    normalize(X)
    cov = dot(transpose(X), X)
    for v in empca.components_:
        u = dot(v, cov)
        assert_almost_parallel(u, v)


if __name__ == '__main__':
    for i in range(1, 10):
        test_eigen_values(random_matrix(20, 10), i)
        print("ok")
