import numpy as np
from sklearn.utils import check_array
from numpy import isnan


def random_matrix(n, m, gen=np.random.uniform):
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            res[i][j] = gen()
    return res


def random_model(N, D, d, missing_p=0):
    W = random_matrix(D, d, gen=lambda : np.random.uniform(-10, 10))
    eps = 0.5

    T = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)
    noise = np.random.multivariate_normal(np.zeros(D), np.eye(D) * eps, N)

    X = T.dot(W.T) + noise

    for i in range(N):
        for j in range(D):
            if np.random.binomial(1, missing_p) == 1:
                X[i][j] = np.nan

    return (X, W.T, T)


def normalize(X):
    mean = []
    for i in range(X.shape[1]):
        cnt = 0
        sum = 0
        for j in range(X.shape[0]):
            if not isnan(X[j][i]):
                cnt += 1
                sum += X[j][i]

        sum /= cnt
        mean.append(sum)

        for j in range(X.shape[0]):
            if not isnan(X[j][i]):
                X[j][i] -= sum

    return np.array(mean)


def submatrix(W, rows):
    A = []
    for i in rows:
        A.append(W[i])
    return np.array(A)


def gram_schmidt(X, tr=False, remove=False, copy=False):
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
        elif not remove:
            Y.append(cur)

    X = np.array(Y)
    if tr:
        X = np.transpose(X)

    return X


def project(v, A):
    """
    Projects vector v on subspace spanned by rows of A
    :param v: vector 1 x D
    :param A: D x d matrix with orhonormal rows
    :return: coordinates of projections of v onto space spanned by rows of A
    """
    u = []
    for col in A:
        u.append(v.dot(col))
    return np.array(u)


def project_many(X, A):
    """
    Projects rows of X onto space spanned by rows of A
    :param X: array-like, n x D
    :param A: D x d matrix with orthonormal rows
    :return: Projections of rows of X onto space spanned by rows of A
    """
    res = []
    for row in X:
        res.append(project(row, A))
    return np.array(res)


def lies_in(v, A):
    for u in A:
        v -= u * v.dot(u)
    return np.linalg.norm(v) <= 1e-1


def span_in(A, B):
    for a in A:
        if not lies_in(a, B):
            return False
    return True


def get_KU(sample):
    k = []
    u = []
    for i in range(len(sample)):
        if isnan(sample[i]):
            u.append(i)
        else:
            k.append(i)
    return k, u


if __name__ == '__main__':
    A = gram_schmidt(np.array([[1, 2],
                               [5, 10]]))

    B = gram_schmidt(np.array([[13, 12],
                               [12, 12312]]))

    print(span_in(A, B))
    exit(0)

    X, W, T = random_model(30, 6, 1)
    print(W)
    exit(0)
    print(random_model(5, 3, 2, 0.1)[0])
    exit(0)
    #Test data
    A = np.array([[0, 1],
                  [-1, 0]])

    x = np.array([1, 1])
    print(project(x, A))
    exit(0)
    test = np.array([[3.0, 1.0], [2.0, 2.0]])
    test2 = np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]])
    print(np.array(gram_schmidt(test)))
    print(np.array(gram_schmidt(test2)))