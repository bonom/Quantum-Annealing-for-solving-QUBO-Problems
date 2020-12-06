#!/usr/bin/env python3

import time
import random
import numpy as np
from scipy import sparse
from QA4QUBO.functions import shuffle_vector, make_decision

def function_f(Q, x):
    return ((np.atleast_2d(x).T).dot(Q)).dot(x)

def shuffle_map(m):
    
    keys = list(m.keys())
    shuffle_vector(keys)
    
    i = 0
    try:
        it = keys[i]
        for key, item in m.items():
            ts = item
            m[key] = m[it]
            m[it] = ts
            i += 1
            try:
                it = keys[i]
            except:
                pass
    except:
        pass


def update(vector):
    dim = len(vector)
    i = 0
    while(i < dim and vector[i] == 1):
        vector[i] = -1
        i += 1
    if(i < dim):
        vector[i] = 1

def minimization_Q(matrix):
    n = len(matrix)
    N = 2**n
    vector = [-1 for i in range(n)]
    minimum = function_f(matrix, np.atleast_2d(vector).T)
    min_vector = vector.copy()
    i = 1
    while (i < N):
        update(vector)
        e = function_f(matrix, np.atleast_2d(vector).T)
        if(e < minimum):
            min_vector = vector.copy()
            minimum = e
        i += 1
    return np.atleast_2d(min_vector).T

def minimization(matrix):
    n = len(matrix)
    matrix = sparse.csr_matrix(matrix)
    rows, cols = matrix.nonzero()
    values = matrix.data
    N = 2**n
    vector = [-1 for i in range(n)]
    
    mat = list(zip(rows,cols,values))
    minimum = E(mat, vector)
    min_vector = vector.copy()
    
    for i in range(N):
        update(vector) 
        e = E(mat, vector)
        if(e < minimum):
            min_vector = vector.copy()
            minimum = e

    return np.atleast_2d(min_vector).T

def E(matrix, vector):
    e = 0
    for row, col, val in matrix:
        if row == col:
            e += val * vector[row]
        else:
            e += val * vector[row] * vector[col]
    return e

def fill(m, perm, n):
    filled = [0 for i in range(n)]
    for i in range(n):
        if i in m:
            filled[i] = perm[m[i]]
        else:
            filled[i] = perm[i]
    return filled


def inverse(perm, n):
    inverted = [0 for i in range(n)]
    for i in range(n):
        inverted[perm[i]] = i

    return inverted


def map_back(z, perm):
    n = len(perm)
    inverted = inverse(perm, n)

    z_ret = [0 for i in range(n)]

    for i in range(n):
        z_ret[i] = z[inverted[i]]

    return z_ret

def g(Q, A, oldperm, pr):
    n = len(Q)
    m = dict()
    for i in range(n):
        if make_decision(pr):
            m[i] = i
            
    shuffle_map(m)

    perm = fill(m, oldperm, n)
    inversed = inverse(perm, n)
    Theta = [[0 for col in range(n)] for row in range(n)]

    for row, col in A:
        k = inversed[row]
        l = inversed[col]
        Theta[row][col] = Q[k][l]#Q.item((k, l))
    return Theta, perm

def h(vect, pr): 
    n = len(vect)
    for i in range(n):
        if make_decision(pr):
            vect[i] = -vect[i]
    return vect

def sim_ann(p, f_prime, f_star):
    if np.log(p) != 0:
        T = -(1/(np.log(p)))
        return np.exp(-(f_prime - f_star)/T)
    return 0

def solve(d_min, eta, i_max, k, lambda_zero, n, N, N_max, p_delta, q, A, Q, make_decision, shuffle_vector):
    check_Q = minimization_Q(Q)
    min_Q = function_f(Q, check_Q).item()
    I = np.identity(n)
    p = 1
    Theta_one, m_one = g(Q, A, np.arange(n), p)
    Theta_two, m_two = g(Q, A, np.arange(n), p)
    
    z_one = map_back(minimization(Theta_one), m_one)
    z_two = map_back(minimization(Theta_two), m_two)
    
    f_one = function_f(Q, z_one).item()
    f_two = function_f(Q, z_two).item()
    if (f_one < f_two):
        z_star = z_one
        f_star = f_one
        m_star = m_one
        z_prime = z_two
    else:
        z_star = z_two
        f_star = f_two
        m_star = m_two
        z_prime = z_one
    if (f_one == f_two) == False:
        S = (np.outer(z_prime, z_prime) - I) + np.diagflat(z_prime)
    else:
        S = [[0 for col in range(n)] for row in range(n)]
    e = 0
    d = 0
    i = 1
    lam = lambda_zero

    sum_time = 0
    
    while True:
        start_time = time.time()
        Q_prime = np.add(Q, (np.multiply(lam, S)))
        if (i % N == 0):
            p = p - ((p - p_delta)*eta)

        Theta_prime, m = g(Q_prime, A, m_star, p)

        z_prime = map_back(minimization(Theta_prime), m)

        if make_decision(q):
            z_prime = h(z_prime, q)

        if (z_prime != z_star):
            f_prime = function_f(Q, z_prime).item()
            if (f_prime < f_star):
                z_prime, z_star = z_star, z_prime
                f_star = f_prime
                m_star = m
                e = 0
                d = 0
                S = S + ((np.outer(z_prime, z_prime) - I) + np.diagflat(z_prime))
            else:
                d = d + 1
                if make_decision(sim_ann(p, f_prime, f_star)):
                    z_prime, z_star = z_star, z_prime
                    f_star = f_prime
                    m_star = m
                    e = 0
            lam = min(lambda_zero, (lambda_zero/(2+(i-1)-e)))
        else:
            e = e + 1

        # debug print
        try:
            print(f"-- -- Valori ciclo {i}/{i_max} -- --\np = {p}, f_prime = {f_prime}, f_star = {f_star}, p**(f_prime-f_star) = {p**(f_prime-f_star)} e = {e}, d = {d} dunque la condizione è (e+d){e+d} >= {N_max}(N_max) and (d){d} < {d_min}(d_min) e lambda = {lam}\nz = {np.atleast_2d(z_star).T}\n  = {np.atleast_2d(check_Q).T}\nCon minimo di Q = {min_Q}\nCi ho messo {time.time()-start_time} secondi\n")
        except:
            print(f"-- -- Ciclo {i}/{i_max} -- --\n\nNon ci sono variazioni\n\nCi ho messo {time.time()-start_time} secondi\n")
        
        sum_time = sum_time + (time.time() - start_time)
        if ((i == i_max) or ((e + d >= N_max) and (d < d_min))):
            print(f"Uscito al ciclo {i}/{i_max} ", end='')
            if(i != i_max):
                print("ed è stata raggiunta la convergenza.")
            else:
                print("\n")
            break
        
        try:
            if(f_prime == min_Q):
                print(f"Found minimum of Q ({min_Q}) equals to f* ({f_star})")
                if input("Should I continue? ['C','c','y','Y','1'] for yes: \n") not in ['C','c','y','Y','1']:
                    break
        except:
            pass
            
        
        i = i + 1
    print(f"Tempo medio per iterazione: {sum_time/i}")

    return np.atleast_2d(np.atleast_2d(z_star).T).T