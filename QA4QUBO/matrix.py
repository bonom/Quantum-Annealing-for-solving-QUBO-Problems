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
    
    #pen = (Q.max() * 2)
    #pen += int(pen/n)
    pen = (Q.max() * 2.25)
    matrix = qubo_qap(P,L,pen)
    #matrix = generate_qubo_model(n, P, L, pen)
    y = pen * (len(P) + len(L))
    return matrix, pen, len(matrix), y
    
def qubo_qap(flow: np.ndarray, distance: np.ndarray, penalty):
    """Quadratic Assignment Problem (QAP)"""
    n = len(flow)
    q = np.einsum("ij,kl->ikjl", flow, distance).astype(np.float)

    i = range(len(q))

    q[i, :, i, :] += penalty
    q[:, i, :, i] += penalty
    q[i, i, i, i] -= 4 * penalty
    return q.reshape(n ** 2, n ** 2)
    
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
    