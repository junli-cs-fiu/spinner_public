#!/usr/bin/env python

from mpi4py import MPI
import sys
import numpy as np
from data import *
from matrix import Carousel
import time

comm = MPI.COMM_WORLD
worldSize = comm.Get_size()
rank = comm.Get_rank()
processorName = MPI.Get_processor_name()

TaskMaster = worldSize - 1
n = int(sys.argv[1])
k = int(sys.argv[2])
D = map(lambda x: int(x), [sys.argv[3], sys.argv[4], sys.argv[5]])
# D = [600, 100000, 1000]
ENC = sys.argv[6]
nof4 = int(sys.argv[7])
name = sys.argv[8]
if ENC != 'RS' and ENC != 'SP1' and ENC != 'SP2' and ENC != "REP" and ENC != "GLO":
    print("invalid encoder")
    exit(1)

import logging

if '-v' in sys.argv:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

if n % 4 != 0:
    print("invalid n")
    exit(1)
if (ENC == "SP1" or ENC == "SP2" or ENC == "GLO") and (nof4 != 0 and nof4 != 4):
    W = [3] * (n - n / 4 * nof4) + [2] * (n / 4 * nof4)
else:
    W = [1] * n
N = sum(W)
R = [0] * n
K = map(lambda x: x * k, W)
if (ENC == 'SP1' or ENC == 'SP2'):
    for i in xrange(1, n):
        R[i] = (R[i - 1] + K[i - 1]) % N

if (D[0] % (k * N) != 0):
    print("invalid D[0]")
    exit(1)
if rank >= n and rank < TaskMaster:
    print("invalid rank")
    exit(1)

logging.debug("[%d] Process %s started." % (rank, processorName))

def worker(rank, Ai, X, I, N, L, D, k, TaskMaster):
    # t_start = MPI.Wtime()
    for i in xrange(I, I + N):
        logging.debug("[%d] Process %d calculating partition %d." % (rank, rank, i % N))
        Y = np.dot(Ai[((i * L) % (D[0] / k)) : (((i + 1) * L) % (D[0] / k))], X)
        request = comm.Ssend(Y, dest = TaskMaster, tag = (i % N))
        # t_comp = MPI.Wtime() - t_start
        # logging.debug("[%d] Process %d finished %d partitions in %5.4fs." %(rank, rank, i-I+1, t_comp))

def master(n, k, D, N, W, ENC):
    t_start = MPI.Wtime()
    M = []
    REQ = []
    for i in xrange(0, n):
        M.append([])
        REQ.append([])
        for j in xrange(N):
            M[i].append(np.zeros((D[0] / (k * N), D[2])))
            # REQ[i].append(comm.Irecv([M[i][j], MPI.FLOAT], source = i, tag = i * N + j))
    helper = []
    for i in xrange(N):
        helper.append([])
    index = []
    Me = np.zeros((k * N, D[0] / (k * N) * D[2]))
    num = 0
    parity = 0

    def print_time():
        time = MPI.Wtime()
        print time - t_start

    if ENC == "SP1" or ENC == "SP2":
        count = 0
        while(not all(len(row) >= k for row in helper)):
            st = MPI.Status()
            comm.Probe(status=st)
            i = st.source
            i1 = i
            j = st.tag
            comm.Recv(M[i1][j], source = i, tag = st.tag)
            if (len(helper[j]) < k):
                if ENC == "SP1":
                    index.append((i1 * N + j, i1 * N + j))
                else:
                    index.append((i1 * N + j, count))
                    count += 1
                    # if (j - R[i1]) % N < k:
                        # index.append((i1 * N + j, i1 * k + ((j - R[i1]) % N)))
                    # else:
                        # index.append((i1 * N + j, k * n + i1 * (n - k) + ((j - R[i1]) % N) - k))
                # Me[num] = M[i1][j].reshape(1, D[0] / (k * N) * D[2])
                # num += 1
                helper[j].append(i1)
                # logging.debug("[%d] worker %d partition %d received" % (rank, i, j))
                # for l in xrange(len(helper)):
                    # logging.debug("[%d] helper on partition %d: [%s]" % (rank, l, " ".join(map(lambda x: str(x), helper[l]))))
                if (ENC == "SP1" and i1 >= k) or (ENC == "SP2" and (j - R[i1]) % N >= K[i1]):
                    parity += 1
        index.sort(key = lambda x: x[1])
        for ind in index:
            i = ind[0] / N
            j = ind[0] % N
            Me[num] = M[i][j].reshape(1, D[0] / (N * k) * D[2])
            num += 1
        index = map(lambda x: x[0], index)
    elif ENC == "RS" or ENC == "REP":
        nodes = [0] * n
        nodes_i = []
        while(len(nodes_i) < k):
            st = MPI.Status()
            comm.Probe(status=st)
            i = st.source
            j = st.tag
            if ENC == "REP":
                i1 = i / (n / k)
                # i1 = k - 1 - i1
            else:
                i1 = i
                # i1 = n - 1 - i1
            comm.Recv([M[i1][j], MPI.FLOAT], source = i, tag = st.tag)
            if i1 not in helper[j]:
                helper[j].append(i1)
                nodes[i1] += 1
                if len(nodes_i) < k and nodes[i1] == N:
                    nodes_i.append(i1)
                logging.debug("[%d] worker %d partition %d received" % (rank, i, j))
                # for l in xrange(len(helper)):
                    # logging.debug("[%d] helper on partition %d: [%s]" % (rank, l, " ".join(map(lambda x: str(x), helper[l]))))
        nodes_i.sort()
        for i in nodes_i:
            if i >= k:
                parity += N
            for j in xrange(N):
                Me[num] = M[i][j].reshape(1, D[0] / (k * N) * D[2])
                num += 1
                index.append(i * N + j)
    elif ENC == "GLO":
        while(len(index) < k * N):
            st = MPI.Status()
            comm.Probe(status=st)
            i = st.source
            i1 = i
            j = st.tag
            comm.Recv([M[i1][j], MPI.FLOAT], source = i, tag = st.tag)
            index.append(i1 * N + j)
            # Me[num] = M[i1][j].reshape(1, D[0] / (k * N) * D[2])
            # num += 1
            logging.debug("[%d] worker %d partition %d received" % (rank, i, j))
            logging.debug("[%d] index received: [%s]" % (rank, " ".join(map(lambda x: str(x), index))))
            if i1 >= k:
                parity += 1
        index.sort()
        for ind in index:
            i = ind / N
            j = ind % N
            Me[num] = M[i][j].reshape(1, D[0] / (k * N) * D[2])
            num += 1
    logging.info("[%d] Parity: %d" % (rank, parity))
    t_data = MPI.Wtime()
    logging.debug("[%d] Enough results obtained." % (rank))
    logging.debug("[%d] Time: %5.4fs" % (rank, MPI.Wtime() - t_start))
    # logging.debug("[%d] helper: [%s]" % (rank, " ".join(map(lambda x: str(len(x)), helper))))
    if ENC == "RS" and parity != 0:
        G, system = RS_plus(n, k, N)
        logging.debug("[%d] Time: %5.4fs" % (rank, MPI.Wtime() - t_start))
        decode(Me, G, index, None)
    elif ENC == "SP1" and parity != 0:
        G, system = RS_plus(n, k, N)
        logging.debug("[%d] Time: %5.4fs" % (rank, MPI.Wtime() - t_start))
        decode(Me, G, index, None)
    elif ENC == "SP2" and parity != 0:
        Ge, system = Carousel(n, k, W)
        logging.debug("[%d] Time: %5.4fs" % (rank, MPI.Wtime() - t_start))
        decode(Me, Ge, index, system)
    elif ENC == "GLO" and parity != 0:
        G, system = Global_RS(n, k, N)
        logging.debug("[%d] Time: %5.4fs" % (rank, MPI.Wtime() - t_start))
        decode(Me, G, index, None)
    t_finish = MPI.Wtime()
    t_all = t_finish - t_start
    t_decode = t_finish - t_data
    logging.info("[%d] Process %d finished in %5.4fs, decoding in %5.4fs" %(rank, rank, t_all, t_decode))
    # return np.array(Me).reshape(D[0], D[2])

if rank != TaskMaster:
    logging.debug("[%d] Running from processor %s (weight = %d), rank %d out of %d processors." % (rank, processorName, W[rank], rank, worldSize))

    # I = sum(K[0 : rank]) % N
    I = R[rank]
    L = D[0] / (k * N)
    logging.debug("[%d] Start Calculation from process %d." % (rank, rank))

    t_start = MPI.Wtime()
    if (ENC == "REP"):
        if (n % k != 0):
            print ("k should be a divisor of n")
            exit(1)
        C = n / k
        # Ai = load("{0}{1}".format(name, rank / C))
        Ai = np.random.rand(D[0] / k, D[1])
        logging.debug("[%d] %d loaded" % (rank, rank / C))
    else:
        # Ai = load("{0}{1}".format(name, rank))
        Ai = np.random.rand(D[0] / k, D[1])
        logging.debug("[%d] %d loaded" % (rank, rank))
    # X = load("{0}X".format(name))
    X = np.random.rand(D[1], D[2])
    comm.Barrier()
    worker(rank, Ai, X, I, N, L, D, k, TaskMaster)
    exit(0)

if rank == TaskMaster:
    logging.debug("[%d] W=[" % (rank) + " ".join(map(lambda x: str(x), W)) + "]")
    logging.debug("[%d] N=%d" % (rank, N))
    logging.debug("[%d] K=[" % (rank) + " ".join(map(lambda x: str(x), K)) + "]")
    logging.debug("[%d] R=[" % (rank) + " ".join(map(lambda x: str(x), R)) + "]")
    logging.debug("[%d] Checking response from Workers." % (rank))

    comm.Barrier()
    master(n, k, D, N, W, ENC)
    comm.Abort(0)
    exit(0)
