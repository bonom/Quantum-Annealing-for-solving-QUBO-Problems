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
import d_wave
import matplotlib.pyplot as plt
import networkx as nx
import sys
np.set_printoptions(threshold=sys.maxsize)


def update(vector):
    dim = len(vector)
    i = 0
    while(i < dim and vector[i] == 1):
        vector[i] = -1
        i += 1
    if(i < dim):
        vector[i] = 1


def print_file(z):
    outF = open("Output.txt", "w")
    for line in z:
        outF.write(str(line))
        outF.write("\n")
    outF.close()


def make_decision(probability):
    return np.random.random() < probability


def function_f(Q, x):
    return ((np.atleast_2d(x).T).dot(Q)).dot(x)


def minimization(matrix):
    n, m = matrix.shape
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


def new_g(A, P, Q, pr):
    n, columns = Q.shape
    m = dict()
    for i in range(n):
        if make_decision(pr):
            m[i] = i
    vals = list(m.values())
    np.random.shuffle(vals)
    m = dict(zip(m.keys(), vals))
    Pprime = np.zeros((n, n))
    final_m = inverse(complete(m,n),n)
    for i in range(n):
        if i in m.values():
            Pprime[i] = P[m[i]]
        else:
            Pprime[i] = P[i]
    Theta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if A.item((i, j)) == 1:
                k = final_m[i]
                l = final_m[j]
                Theta[i][j] = Q[k][l]
    return Pprime, Theta


def inverse(dic,n):
    inverted = dict()
    for i in range(n):
        inverted[dic[i]] = i
    return inverted


def complete(dic, n):
    filled = dict()
    for i in range(n):
        if(i in dic.keys()):
            filled[i] = dic[i]
        else:
            filled[i] = i
    return filled


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
    n, m = vect.shape
    for i in range(n):
        if make_decision(pr):
            vect[i] = -vect[i]
    return vect


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
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
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
            break
    print(f"\n--- Uscito al ciclo {i} ---\n{z_star}")
    return z_star


def QALS_gless(d_min, eta, i_max, k, lambda_zero, n, N_max, p_delta, q, A, Q):
    I = np.identity(n)
    P = I
    p = 1
    P_one, Theta_one = new_g(A, P, Q, 1)
    P_two, Theta_two = new_g(A, P, Q, 1)
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
        P, Theta_prime = new_g(A, P_star, Q_prime, p)
        # for i in range(k):
        #z_prime = (np.transpose(P)).dot(random_z(n))
        z_prime = (np.transpose(P)).dot(minimization(Theta_prime))
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
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
            break
    print(f"\n--- Uscito al ciclo {i} ---\n{z_star}")
    return z_star


def show_graph(adjacency_matrix, mylabels=None):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    plt.show()


def main():
    """Dati """
    i_max = 1000
    q = 0.01
    p_delta = 0.1
    eta = 0.1
    lambda_zero = 1
    k = 20
    N_max = 100
    d_min = 100
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
    Q = np.zeros((n, n))
    for i in range(n):
        j_max += 1
        while j < j_max:
            Q[i][j] = np.random.randint(low=-10, high=10)
            Q[j][i] = Q[i][j]
            j += 1
        j = 0
    print(f"FATTO!\n--------------- Q matrice {Q.shape} ---------------\n{Q}")
    print(f"\nCreo A ...", end=' ')
    """
    j_max = 0
    j = 0
    matrix_A = np.zeros((n, n))
    for i in range(n):
        j_max += 1
        while j < j_max:
            matrix_A[i][j] = np.random.randint(2)
            matrix_A[j][i] = matrix_A[i][j]
            j += 1
        j = 0
    print(
        f"FATTO!\n--------------- A matrice {A.shape} ---------------\n{A}\n")
    """ 
    if(n % 8 == 0) and (rows * columns * 8 == n):
        A = d_wave.chimera_graph(rows, columns)
        matrix_A = (nx.adjacency_matrix(A)).todense()
    else:
        exit("Error", -1)
    
    print(
        f"FATTO!\n--------------- A matrice {matrix_A.shape} ---------------\n{matrix_A}\n")
    if(input("Vuoi vedere il grafo di A (S/n)? ") in ["S", "s", "y", "Y"]):
        show_graph(matrix_A)
        # d_wave.print_graph(A)

    print("\n")
    #start_time = time.time()
    #
    #QALS(d_min, eta, i_max, k, lambda_zero, n, N_max, p_delta, q, matrix_A, Q)
    #
    #print("\n------------ Impiegati %s secondi con la g normale ------------\n\n" %(time.time() - start_time))
    start_time = time.time()
    QALS_gless(d_min, eta, i_max, k, lambda_zero, n, N_max, p_delta, q, matrix_A, Q)

    # print_file(z)
    print("\n------------ Impiegati %s secondi con una gless ------------\n\n" %
          (time.time() - start_time))


if __name__ == "__main__":
    main()
