#!/usr/bin/env python3

import dwave_networkx as dnx
import networkx as nx
import numpy as np
import dimod

def generate_QUBO_problem(S):
    """
        Generate a QUBO problem (The number partitioning problem) from a vector S
    """
    n = len(S)
    c = 0
    for i in range(n):
        c += S[i]
    col_max = 0
    col = 0
    QUBO = [[0 for col in range(n)] for row in range(n)]
    for row in range(n):
        col_max += 1
        while col < col_max:
            if row == col:
                QUBO[row][col] = S[row]*(S[row]-c)
            else:
                QUBO[row][col] = S[row] * S[col]
                QUBO[col][row] = QUBO[row][col]
            col += 1
        col = 0
    return QUBO, c

def read_integers(filename:str):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]

def generate_QAP_problem(file):
    file_it = iter(read_integers(file))
    n = next(file_it)
    P = [[next(file_it) for j in range(n)] for i in range(n)]
    L = [[next(file_it) for j in range(n)] for i in range(n)]
    
    Q = np.kron(P,L)
    """
    std_dev = 0
    count = 0
    mean = np.mean(Q)
    for i in range(len(Q)):
        for j in range(len(Q)):
            if(Q[i][j] != 0):
                std_dev += ((Q[i][j]- mean) ** 2)
                count += 1
   
    std_dev /= count - 1
    """
    #pen = (Q.max() + np.sqrt(std_dev)*2.25)
    pen = Q.max() * 2.25
    matrix = generate_qubo_model(n, P, L, pen)
    y = pen * (len(P) + len(L))
    return matrix, pen, len(matrix), y
"""

def generate_QAP_problem(file):
    file_it = iter(read_integers(file))
    n = next(file_it)
    P = [[next(file_it) for j in range(n)] for i in range(n)]
    L = [[next(file_it) for j in range(n)] for i in range(n)]

   
    
    return qubo_qap(P,L,pen)
"""
def qubo_qap(flow: np.ndarray, distance: np.ndarray, penalty):
    """Quadratic Assignment Problem (QAP)"""
    n = len(flow)
    q = np.einsum("ij,kl->ikjl", flow, distance).astype(np.float)

    i = range(len(q))

    q[i, :, i, :] += penalty
    q[:, i, :, i] += penalty
    q[i, i, i, i] -= 4 * penalty
    return q.reshape(n ** 2, n ** 2)

def generate_qubo_model(n, A, B, P=None):
    
    # The Q matrix is initialized
    Q = np.zeros(shape=(n*n,n*n))
    
    if P is None:
        P = max(map(max, A)) * max(map(max, B))/2
    offset = 2*n*P
    
    # The Q matrix is filled
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    # These correspond to the diagonal of the matrix
                    if i==j and k==l:
                        Q[i*n+k, j*n+l] = -2*P
                    # These correspond to the decision variables that can't occur at the same time, which is
                    # when a facility is in two locations or two facilities are in the same location
                    elif i==j or k==l:
                        Q[i*n+k, j*n+l] = P
                    # Valid pairs of decision variables come directly from the original objective function
                    else:
                        Q[i*n+k, j*n+l] = (A[i][j] * B[k][l]) / 2

    return Q
    
def generate_chimera(n):
    G = dnx.chimera_graph(16)
    tmp = nx.to_dict_of_lists(G)
    rows = []
    cols = []
    for i in range(n):
        rows.append(i)
        cols.append(i)
        for j in tmp[i]:
            if(j < n):
                rows.append(i)
                cols.append(j)

    return list(zip(rows, cols))

def generate_pegasus(n):
    G = dnx.pegasus_graph(16)

    tmp = nx.to_numpy_matrix(G)
    
    rows = []
    cols = []
           
    for i in range(n):
        rows.append(i)
        cols.append(i)
        for j in range(n):
            if(tmp.item(i,j)):
                rows.append(i)
                cols.append(j)

    return list(zip(rows, cols))
    