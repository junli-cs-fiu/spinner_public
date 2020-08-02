import numpy as np

def Cauchy(m, n):
    x = np.array(xrange(n + 1, n + m + 1))
    y = np.array(xrange(1, n + 1))
    x = x.reshape((-1, 1))
    diff_matrix = x - y
    cauchym = 1.0 / diff_matrix
    return cauchym


def RS(n, k):
    I = np.identity(k)
    P = Cauchy(n - k, k)
    return np.concatenate((I, P), axis = 0)


def RS_plus(n, k, N):
    G = RS(n, k)
    Ge = np.zeros((n * N, k * N))
    for i in xrange(k * N):
        Ge[i, i] = 1
    for i in xrange(k, n):
        for j in xrange(k):
            for l in xrange(N):
                Ge[i * N + l, j * N + l] = G[i, j]
    return (Ge, list(xrange(0, k * N)))


def Global_RS(n, k, N):
    G = RS(n * N, k * N)
    return (G, list(xrange(0, k * N)))


def Carousel(n, k, W):
    N = sum(W)
    K = map(lambda x: x * k, W)
    R = [0] * n
    for i in xrange(1, n):
        R[i] = (R[i - 1] + K[i - 1]) % N

    Ge = RS_plus(n, k, N)[0]
    index = [0] * (k * N)
    count = 0
    for i in xrange(n):
        for j in xrange(K[i]):
            index[count] = i * N + (R[i] + j) % N
            count += 1

    Ge = np.linalg.solve(Ge[index].T, Ge.T)
    Ge = Ge.T
    return (Ge, index)


if __name__ == "__main__":
    print Cauchy(4,3)
    print RS(5, 4)
