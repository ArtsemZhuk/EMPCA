"""Expectation-Maximization for Principal Component Analysis with missing values"""
# Author: Artsem Zhuk


import numpy as np
import sys
from numpy.linalg import inv
from numpy import dot, transpose, trace
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from utils import gram_schmidt, normalize, submatrix


class EMPCAM(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, n_iter=100):
        self.n_components = n_components
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

        self.mean_ = normalize(X)

        N, D = X.shape
        d = self.n_components

        K = []
        U = []

        for sample in X:
            k = []
            u = []
            for i in range(D):
                if sample[i] is None:
                    u.append(i)
                else:
                    k.append(i)
            K.append(k)
            U.append(u)


        sigma = 1.0
        W = np.zeros([D, d], dtype=np.float64)
        for i in range(D):
            for j in range(d):
                W[i][j] = np.random.uniform(-1, 1)

        for _ in range(self.n_iter):
            assert W.shape == (D, d)
            # e-step

            Wu = []
            Wk = []
            Wkt = []
            Wut = []
            x = []
            xk = []
            for i in range(N):
                Wu.append(submatrix(W, U[i]))
                Wk.append(submatrix(W, K[i]))
                Wut.append(transpose(Wu[i]))
                Wkt.append(transpose(Wk[i]))

                x.append(transpose(np.array([X[i]])))
                xk.append(submatrix(x[i], K[i]))
                xk[i] = np.array(xk[i], dtype=np.float64)
                assert xk[i].shape == (len(K[i]), 1)


            M = []
            for i in range(N):
                Mi = dot(Wkt[i], Wk[i]) + sigma * np.eye(d)
                Mi = inv(Mi)
                assert Mi.shape == (d, d)
                M.append(Mi)
            assert len(M) == N

            S = []
            for i in range(N):
                if not U[i]:
                    S.append([[[], []],
                              [[], sigma * M[i]]])
                else:
                    A = np.eye(len(U[i])) + Wu[i].dot(M[i]).dot(Wut[i])
                    A *= sigma
                    assert A.shape == (len(U[i]), len(U[i]))

                    B = -1 * Wu[i].dot(M[i])
                    B *= sigma
                    assert B.shape == (len(U[i]), d)

                    C = transpose(B)

                    S.append([[A, B],
                              [C, M[i] * sigma]])
            assert len(S) == N

            m = []
            for i in range(N):
                r = dot(Wkt[i], xk[i])
                r = dot(M[i], r)

                if U[i]:
                    l = dot(Wu[i], r)
                    l = transpose(l)[0]
                else:
                    l = []

                r = transpose(r)[0]
                m.append((l, r))
                assert len(m[i][1]) == d

            assert len(m) == N


            # m-step
            ETT = np.zeros([d, d], dtype=np.float64)
            for i in range(N):
                ETT += S[i][1][1]

            W_new = np.zeros([D, d], dtype=np.float64)
            for n in range(N):
                cnt = 0
                for i in range(D):
                    if X[n][i] is not None:
                        W_new[i] += X[n][i] * m[n][1]
                    else:
                        W_new[i] += S[n][0][1][cnt]
                        cnt += 1

            W_new = W_new.dot(inv(ETT))
            W_new = gram_schmidt(W_new, tr=True)

            sigma_new = 0

            for n in range(N):
                cnt = 0
                for i in range(D):
                    if X[n][i] is not None:
                        sigma_new += np.square(X[n][i])
                    else:
                        sigma_new += np.square(m[n][0][cnt]) + S[n][0][0][cnt][cnt]

                et = m[n][1]
                et = dot(et, Wkt[n]).dot(xk[n])
                sigma_new -= 2 * np.float64(et)

                if U[n]:
                    sigma_new -= 2 * trace(Wut[n].dot(S[n][0][1]))

                sigma_new += trace(Wkt[n].dot(Wk[n]).dot(S[n][1][1]))

                if U[n]:
                    sigma_new += trace(Wut[n].dot(Wu[n]).dot(S[n][1][1]))



            sigma_new /= N * D
            print(sigma_new)
            print(W_new)

            W = W_new
            sigma = sigma_new


        self.components_ = gram_schmidt(transpose(W))
        self.sigma_ = sigma


if __name__ == '__main__':
    X = np.array(
        [[1, 2, None],
         [None, None, 3],
         [10, 20, 30]],
    )

    e = EMPCAM(n_components=2, n_iter=200)
    e.fit(X)