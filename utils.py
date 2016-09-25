import numpy as np
from sklearn.utils import check_array

def normalize(X):
    mean = []
    for i in range(X.shape[1]):
        cnt = 0
        sum = 0
        for j in range(X.shape[0]):
            if X[j][i] is not None:
                cnt += 1
                sum += X[j][i]

        sum /= cnt
        mean.append(sum)

        for j in range(X.shape[0]):
            if X[j][i] is not None:
                X[j][i] -= sum

    return np.array(mean)


def submatrix(W, rows):
    A = []
    for i in rows:
        A.append(W[i])
    return np.array(A)


def gram_schmidt(X, tr=False, copy=False):
    X = check_array(X, dtype=[np.float64], ensure_2d=True, copy=copy)

    if tr:
        X = np.transpose(X)

    Y = []

    for cur in X:
        for y in Y:
            cur = cur - y * np.dot(cur, y)

        len = np.linalg.norm(cur)
        if len > 0:
            Y.append(cur / len)

    X = np.array(Y)
    if tr:
        X = np.transpose(X)

    return X


if __name__ == '__main__':
    #Test data
    test = np.array([[3.0, 1.0], [2.0, 2.0]])
    test2 = np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]])
    print(np.array(gram_schmidt(test)))
    print(np.array(gram_schmidt(test2)))