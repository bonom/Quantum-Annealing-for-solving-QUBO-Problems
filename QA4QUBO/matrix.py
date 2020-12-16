#!/usr/bin/env python3

import dwave_networkx as dnx
import networkx as nx
import numpy as np

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

def generate_QAP_problem(file):
    n = int(file.readline())
    tmp = file.readlines()
    for element in tmp:
        element = element.rstrip('\n')
        
    P = [[0 for i in range(n)] for j in range(n)]
    L = [[0 for i in range(n)] for j in range(n)]
    for i in range(1, n):
        row_s = tmp[i].split(' ')
        row_s[len(row_s)-1] = row_s[len(row_s)-1].rstrip("\n")
        row = [int(element) for element in row_s]
        P[i] = row

    for i in range(1, n):
        row_s = tmp[i].split(' ')
        row_s[len(row_s)-1] = row_s[len(row_s)-1].rstrip("\n")
        row = [int(element) for element in row_s]
        L[i] = row

    return qubo_qap(P,L)

def qubo_qap(flow: np.ndarray, distance: np.ndarray, penalty=10.):
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
    G = dnx.pegasus_graph(16, fabric_only=False)
    
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
