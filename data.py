import numpy as np
from matrix import *
import logging


def matrix_generator(X, Y):
    M = np.random.rand(X,Y)
    return M


def encode_RS(M, k, r):
    G = RS(k + r, k)
    M = np.split(M, k)
    return multiply(M, G)


def decode_RS(M, k, r, index):
    G = RS(k + r, k)
    G = G[index]
    # print G
    return multiply(M, np.linalg.inv(G))


def encode_RS_i(M, k, r, index):
    G = RS(k + r, k)
    G = G[index]
    return multiply(M, G)


def encode_Carousel_i(M, k, r, W, index):
    N = sum(W)
    G = Carousel(k + r, k, W)
    G = G[index * N, (index + 1) * N]
    Me = multiply(M, G)
    return np.concatenate(Me, axis = 0)


def encode(M, G):
    N, K = G.shape
    M = np.split(M, K)
    return multiply(M, G)


def decode(M, G, index, system = None):
    G = G[index]
    if system != None:
        G = np.linalg.inv(G)
        todo = []
        for i in xrange(len(system)):
            if not(system[i] in index):
                todo.append(i)
        G = G[todo]
        # print G.shape
        # print np.linalg.matrix_rank(G)
        return multiply(M, G)
    else:
    # print np.size(G, 0), np.size(G, 1)
    # print np.size(M, 0), np.size(M, 1)
        X = np.linalg.solve(G, M)
        return X


def multiply(M, G):
    count = 0
    D = M[0].shape
    X = 1
    Y = D[-1]
    N, K = G.shape
    R = np.zeros((N, X, Y))
    for i in xrange(N):
        # print G[i]
        for j in xrange(K):
            if G[i, j] != 0:
                R[i] = R[i] + G[i, j] * M[j]
                count += 1
    logging.info(" matrix multiplication: %d" % (count,))
    return R


def load(name):
    return np.loadtxt(name)


def save(M, name):
    np.savetxt(name, M)


if __name__ == "__main__":
    import sys
    name = sys.argv[1]
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    r = n - k
    D = map(lambda x: int(x), [sys.argv[4], sys.argv[5], sys.argv[6]])
    ENC = sys.argv[7]
    if ENC != 'RS' and ENC != 'SP1' and ENC != 'SP2' and ENC != 'REP' and ENC != "GLO":
        exit(1)
    W = [1] * n
    N = 1
    R = [0] * n
    K = map(lambda x: x * k, W)
    if (ENC == 'SP1' or ENC == 'SP2' or ENC == "GLO"):
        N = sum(W)
        for i in xrange(1, n):
            R[i] = (R[i - 1] + K[i - 1]) % N
    if (D[0] % N != 0):
        exit(1)

    M = matrix_generator(D[0], D[1])
    M0 = M
    if (ENC == 'RS' or ENC == 'SP1'):
        M = encode_RS(M, k, r)
        for i in xrange(k + r):
            save(M[i], "{0}{1}".format(name, i))
        X = matrix_generator(D[1], D[2])
    elif (ENC == 'SP2'):
        G = Carousel(n, k, W)
        M = encode(M, G)
        for i in xrange(k + r):
            save(np.concatenate(M[i * N : (i + 1) * N], axis = 0), "{0}{1}".format(name, i))
        X = matrix_generator(D[1], D[2])
        save(X, "{0}X".format(name))
    elif (ENC == "GLO"):
        G = Global_RS(n, k, N)
        M = encode(M, G)
        for i in xrange(k + r):
            save(np.concatenate(M[i * N : (i + 1) * N], axis = 0), "{0}{1}".format(name, i))
        X = matrix_generator(D[1], D[2])
        save(X, "{0}X".format(name))
    elif (ENC == "REP"):
        if (n % k != 0):
            exit(1)
        C = n / k
        M = encode_RS(M, k, k)
        for i in xrange(k):
            save(M[i], "{0}{1}".format(name, i))
        X = matrix_generator(D[1], D[2])
        save(X, "{0}X".format(name))
    DEBUG = False
    if (DEBUG):
        if ENC == "RS" or ENC == "SP1":
            F = M[r : k + r]
            F = decode_RS(F, k, r, xrange(r, k + r))
        elif ENC == "SP2":
            F = M[r * N : (k + r) * N]
            F = decode(F, G, xrange(r * N, (k + r) * N))
        else:
            exit(0)
        F = np.array(F).reshape(D[0],D[1])
        # print (F - M0).reshape(D[0],D[1])
        print np.linalg.norm(F - M0, 1)
