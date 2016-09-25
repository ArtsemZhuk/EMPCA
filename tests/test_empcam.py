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
    print(gram_schmidt(W.T))


if __name__ == '__main__':
    show(100, 5, 2)
