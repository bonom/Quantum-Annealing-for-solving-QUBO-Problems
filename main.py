"""Dati utili al problema"""

"""
    i_max = 1000
    q = 0.01
    p_delta = 0.1
    eta = 0.1
    lambda_zero = 1 (per ora)
    k = 20
    N_max = 100
    d_min = 100
"""

import numpy as np
import time
import dwave_networkx as dnx
import networkx as nx
import sys
import annealer

def update(vector):
    dim = len(vector)
    i = 0
    while(i < dim and vector[i] == 1):
        vector[i] = -1
        i += 1
    if(i < dim):
        vector[i] = 1

def make_decision(probability):
    return np.random.random() < probability


def function_f(Q, x):
    return ((np.atleast_2d(x).T).dot(Q)).dot(x)


def minimization(matrix):
    n = len(matrix)
    N = 2**n
    vector = np.empty([n])
    for i in range(n):
        vector[i] = -1
    minimum = function_f(matrix, np.atleast_2d(vector).T)
    min_vector = vector.copy()
    i = 1
    while (i < N):
        print(f"Minimizzazione in corso...  {int((i/N)*100)}%", end="\r")
        update(vector)
        e = function_f(matrix, np.atleast_2d(vector).T)
        if(abs(e) < abs(minimum)):
            min_vector = vector.copy()
            minimum = e
        i += 1
    return np.atleast_2d(min_vector).T

def g(P, pr):
    row, col = P.shape
    Pprime = np.zeros((row, col))
    m = dict()
    for i in range(row):
        if make_decision(pr):
            m[i] = i
    vals = list(m.values())
    np.random.shuffle(vals)
    m = dict(zip(m.keys(), vals))
    for i in range(row):
        if i in m.values():
            Pprime[i] = P[m[i]]
        else:
            Pprime[i] = P[i]
    return Pprime


def h(vect, pr):  # algorithm 4
    n = len(vect)
    for i in range(n):
        if make_decision(pr):
            vect[i] = -vect[i]
    return vect

def g_faster(Q, A, oldperm, pr):#, P):
    n = len(Q)
    m = dict()
    for i in range(n):
        if make_decision(pr):
            m[i] = i
    vals = list(m.values())
    np.random.shuffle(vals)
    m = dict(zip(m.keys(), vals))
    """
    Pprime = np.empty((n, n), dtype=int)
    for i in range(n):
        if i in m.values():
            Pprime[i] = P[m[i]]
        else:
            Pprime[i] = P[i]
    """
    perm = fill(m, oldperm, n)
    inversed = inverse(perm, n)

    Theta = np.zeros((n, n))

    rows,cols = A.nonzero()
    for row,col in zip(rows,cols):
        k = inversed[row]
        l = inversed[col]
        Theta[row][col] = Q.item((k,l))
    
    return Theta, perm


def fill(m, perm, n):
    filled = np.empty((n), dtype=int)
    for i in range(n):
        if i in m:
            filled[i] = perm[m[i]]
        else:
            filled[i] = perm[i]
    return filled


def inverse(perm, n):
    inverted = np.empty((n), dtype=int)
    for i in range(n):
        inverted[perm[i]] = i

    return inverted

def map_back(z, perm):
    n = len(perm)
    inverted = inverse(perm, n)

    z_ret = np.empty((n))

    for i in range(n):
        z_ret[i] = z[inverted[i]]

    return np.atleast_2d(z_ret).T


def QALS(d_min, eta, i_max, k, lambda_zero, n, N_max, p_delta, q, A, Q):
    I = np.identity(n)
    P = I
    p = 1
    P_one = g(P, 1)
    P_two = g(P, 1)
    Theta_one = np.multiply((((np.transpose(P_one)).dot(Q)).dot(P_one)), (A))
    Theta_two = np.multiply((((np.transpose(P_two)).dot(Q)).dot(P_two)), (A))
    # for i in range(k):
    #z_one = (np.transpose(P_one)).dot(random_z(n))
    #z_two = (np.transpose(P_one)).dot(random_z(n))
    z_one = (np.transpose(P_one)).dot(minimization(Theta_one))
    z_two = (np.transpose(P_two)).dot(minimization(Theta_two))
    f_one = function_f(Q, z_one)
    f_two = function_f(Q, z_two)
    if (f_one < f_two).all():
        z_star = z_one
        f_star = f_one
        P_star = P_one
        z_prime = z_two
    else:
        z_star = z_two
        f_star = f_two
        P_star = P_two
        z_prime = z_one
    if (f_one == f_two).all() == False:
        S = (np.outer(z_prime, z_prime) - I) + np.diagflat(z_prime)
    else:
        S = np.zeros((n, n))
    e = 0
    d = 0
    i = 0
    lam = lambda_zero
    while True:
        print(f"-- Ciclo numero {i + 1}")  # , end = "\r")
        Q_prime = np.add(Q, (np.multiply(lam, S)))
        if (i % n == 0):
            p = p - ((p - p_delta)*eta)
        P = g(P_star, p)
        Theta_prime = np.multiply(
            (((np.transpose(P)).dot(Q_prime)).dot(P)), (A))
        # for i in range(k):
        #z_prime = (np.transpose(P)).dot(random_z(n))
        z_prime = (np.transpose(P)).dot(minimization(Theta_prime))
        sys.stdout.write("\033[K\033[F\033[K")
        if make_decision(q):
            z_prime = h(z_prime, q)
        if (z_prime == z_star).all() == False:
            f_prime = function_f(Q, z_prime)
            if (f_prime < f_star).all():
                z_prime, z_star = z_star, z_prime
                f_star = f_prime
                P_star = P
                e = 0
                d = 0
                S = S + ((np.outer(z_prime, z_prime) - I) +
                         np.diagflat(z_prime))
            else:
                d = d + 1
                if make_decision((p**(f_prime-f_star))):
                    z_prime, z_star = z_star, z_prime
                    f_star = f_prime
                    P_star = P
                    e = 0
            # R:37 lambda diminuirebbe
            # lam = lam - i/i_max
        else:
            e = e + 1
        i = i + 1
        if ((i == i_max) or ((e + d >= N_max) and (d < d_min))):
            sys.stdout.write("\033[K")
            print(f"Uscito al ciclo {i}/{i_max} ", end = '')
            if(i != i_max):
                print("ed è stata raggiunta la convergenza.")
            else:
                print("\n")
            break
        
    return z_star


def QALS_g(d_min, eta, i_max, k, lambda_zero, n, N_max, p_delta, q, A, Q):
    I = np.identity(n)
    P = I
    p = 1
    Theta_one, m_one = g_faster(Q, A, np.arange(n), p)
    Theta_two, m_two = g_faster(Q, A, np.arange(n), p)
    # for i in range(k):
    #z_one = (np.transpose(P_one)).dot(minimization(Theta_one))
    #z_two = (np.transpose(P_two)).dot(minimization(Theta_two))
    z_one = map_back(minimization(Theta_one), m_one)
    z_two = map_back(minimization(Theta_two), m_two)
    f_one = function_f(Q, z_one).item()
    f_two = function_f(Q, z_two).item()
    if (f_one < f_two):
        z_star = z_one
        f_star = f_one
        #P_star = P_one
        m_star = m_one
        z_prime = z_two
    else:
        z_star = z_two
        f_star = f_two
        #P_star = P_two
        m_star = m_two
        z_prime = z_one
    if (f_one == f_two) == False:
        S = (np.outer(z_prime, z_prime) - I) + np.diagflat(z_prime)
    else:
        S = np.zeros((n, n))
    e = 0
    d = 0
    i = 0
    lam = lambda_zero

    sum_time = 0

    while True:
        start_time = time.time()
        print(f"-- Ciclo numero {i + 1}")  # , end = "\r")
        Q_prime = np.add(Q, (np.multiply(lam, S)))
        if (i % n == 0):
            p = p - ((p - p_delta)*eta)

        Theta_prime, m = g_faster(Q_prime, A, m_star, p)
        # for i in range(k):
        #z_prime = (np.transpose(P)).dot(minimization(Theta_prime))
        z_prime = map_back(minimization(Theta_prime), m)
        sys.stdout.write("\033[K\033[F\033[K")
        if make_decision(q):
            z_prime = h(z_prime, q)
        if (z_prime == z_star).all() == False:
            f_prime = function_f(Q_prime, z_prime).item()
            #print(f"f_prime = {f_prime} con {type(f_prime)}")
            if (f_prime < f_star):
                z_prime, z_star = z_star, z_prime
                f_star = f_prime
                #P_star = P
                m_star = m
                e = 0
                d = 0
                S = S + ((np.outer(z_prime, z_prime) - I) +
                         np.diagflat(z_prime))
            else:
                d = d + 1
                #print(f"cosa è p : {type(p)} con valore {p}\ncosa è f_prime : {type(f_prime)} con valore {f_prime}\ncosa è f_star {type(f_star)} con valore {f_star}")
                if make_decision((p**(f_prime-f_star))):
                    z_prime, z_star = z_star, z_prime
                    f_star = f_prime
                    #P_star = P
                    m_star = m
                    e = 0
            # R:37 lambda diminuirebbe
            # lam = lam - i/i_max
        else:
            e = e + 1
        i = i + 1
        print(f"-- --- Valori -- --\np = {p}, f_prime = {f_prime}, f_star = {f_star} e z\n{z_star}")
        if ((i == i_max) or ((e + d >= N_max) and (d < d_min))):
            sys.stdout.write("\033[K")
            print(f"Uscito al ciclo {i}/{i_max} ", end = '')
            if(i != i_max):
                print("ed è stata raggiunta la convergenza.")
            else:
                print("\n")
            break
        sum_time += (time.time() - start_time)
    
    print(f"Tempo medio per iterazione: {sum_time/i}")

    return z_star

def main():
    """Dati """
    i_max = 3000
    q = 0.1
    p_delta = 0.1
    eta = 0.01
    lambda_zero = 1.0
    k = 1
    N_max = 50
    d_min = 30
    n = 16
    """
    Solo per test
    """
    rows = 1
    columns = 2

    """MAIN"""
    print(f"Creo Q ...", end=' ')
    
    j_max = 0
    j = 0
    #Q = np.zeros((n, n))
    #for i in range(n):
    #    j_max += 1
    #    while j < j_max:
    #        Q[i][j] = np.random.randint(low=-10, high=10)
    #        Q[j][i] = Q[i][j]
    #        j += 1
    #    j = 0

    Q = np.matrix([
         [0,  0.2,  9.7,  9.5, -4.6, -8.2,  4.7, -3.4, -3.5,    6,  0.2,  6.3,  8.6,  8.7,  7.8,  3.9],
       [0.2,    1, -0.1, -7.1, -5.5,  9.3,  6.8, -7.7,  6.7,   -3, -3.5, -6.9,    9,  4.6, -6.2, -0.4],
       [9.7, -0.1,    2, -1.9, -0.3,  9.8, -2.3, -0.9, -9.9, -5.5, -6.2, -3.2,    1,  4.8,    2, -7.7],
       [9.5, -7.1, -1.9,    3,  3.3,  0.7,  5.1, -7.9, -9.5,    8, -8.3,  9.8,  4.7,  9.1,  6.4,  6.7],
      [-4.6, -5.5, -0.3,  3.3,    4, -4.5,  4.6,  5.7,  -10,  3.4, -4.8, -1.9,   -7,   10,  0.8,  2.2],
      [-8.2,  9.3,  9.8,  0.7, -4.5,    5, -9.9,  5.4,    6, -8.1, -8.7,  0.7,  3.9, -6.4,    9, -5.5],
       [4.7,  6.8, -2.3,  5.1,  4.6, -9.9,    6,  3.7, -8.9,  -10,  1.6,  7.9,  4.8, -8.8,  6.9,  1.2],
      [-3.4, -7.7, -0.9, -7.9,  5.7,  5.4,  3.7,    7,    8, -7.7, -9.3, -1.4,  7.4,  4.1,  3.8, -9.5],
      [-3.5,  6.7, -9.9, -9.5,  -10,    6, -8.9,    8,    8,  2.1,  3.7,  1.3, -5.8, -1.2, -8.4,  5.2],
         [6,   -3, -5.5,    8,  3.4, -8.1,  -10, -7.7,  2.1,    9,  0.7,  8.1, -4.2,  9.7,  6.7,  9.9],
       [0.2, -3.5, -6.2, -8.3, -4.8, -8.7,  1.6, -9.3,  3.7,  0.7,   10,  9.2,  0.4,    6,  9.3,    7],
       [6.3, -6.9, -3.2,  9.8, -1.9,  0.7,  7.9, -1.4,  1.3,  8.1,  9.2,   11,  3.8,    4,  8.3,  0.6],
       [8.6,    9,    1,  4.7,   -7,  3.9,  4.8,  7.4, -5.8, -4.2,  0.4,  3.8,   12, -9.9,  1.2, -2.1],
       [8.7,  4.6,  4.8,  9.1,   10, -6.4, -8.8,  4.1, -1.2,  9.7,    6,    4, -9.9,   13,  5.9,  9.8],
       [7.8, -6.2,    2,  6.4,  0.8,    9,  6.9,  3.8, -8.4,  6.7,  9.3,  8.3,  1.2,  5.9,   14, -9.8],
       [3.9, -0.4, -7.7,  6.7,  2.2, -5.5,  1.2, -9.5,  5.2,  9.9,    7,  0.6, -2.1,  9.8, -9.8,   15]])

    print(f"FATTO!\n--------------- Q matrice {Q.shape} ---------------\n{Q}")
    print(f"\nCreo A ...", end=' ')
    
    if(rows * columns * 8 == n):
        A = dnx.chimera_graph(rows, columns)
        matrix_A = nx.adjacency_matrix(A)#.todense()
    else:
        exit("Error", -1)
    
    print(
        f"FATTO!\n--------------- A matrice {matrix_A.shape} ---------------\n{matrix_A.todense()}\n")

    start_time = time.time()
    print(f"Dati inseriti:\nd min = {d_min}\neta = {eta}\ni max = {i_max}\nk = {k}\nlambda zero = {lambda_zero}\nn = {n}\nN max = {N_max}\np delta = {p_delta}\nq = {q}\n")
    QALS_g(d_min, eta, i_max, k, lambda_zero, n, N_max, p_delta, q, matrix_A, Q)
    
    print("\n------------ Impiegati %0.2f secondi con l'algoritmo ------------\n\n" %
          (time.time() - start_time))


if __name__ == "__main__":
    main()
