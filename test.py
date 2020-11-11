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
import matplotlib.pyplot as plt
import networkx as nx
import sys
import hybrid
np.set_printoptions(threshold=sys.maxsize)


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


def h(vect, pr):  # algorithm 4
    n, m = vect.shape
    for i in range(n):
        if make_decision(pr):
            vect[i] = -vect[i]
    return vect

def check_equality(A,B):
    row_A, col_A = A.shape
    row_B, col_B = B.shape
    if row_A == row_B and col_A == col_B:
        for row in range(row_A):
            for col in range(col_A):
                if A.item((row,col)) != B.item((row,col)):
                    return False
        return True
    else:
        return False
    


def new_g(A, P, Q, old_dic, pr):
    print(f"Viene passata P come:\n{P}\n -- e Q come --\n{Q}")
    print(f"Old_dic = {old_dic} e pr = {pr}")
    n, columns = Q.shape
    m = np.arange((n))
    for i in range(n):
        if make_decision(pr):
            m[i] = i
            
    np.random.shuffle(m)
    tmp = complete(m, old_dic, n)
    final_m = inverse(tmp, n)
    
    #Devo farlo per test e per tornare la matrice per le operazioni di z_star
    Pprime = np.zeros((n, n))
    print(f"m = {m} vettore posizioni = {final_m}")
    for i in range(n):
        if i in m:
            Pprime[i] = P[m[i]]
        else:
            Pprime[i] = P[i]
    
    print(f"Ne esce un Pprime così: \n{Pprime}")
    Theta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if A.item((i, j)) == 1:
                print(f"A[{i}][{j}] == 1")
                k = final_m[i]
                print(f"k = final_m[{i}] -- {k} = {final_m[i]}")
                l = final_m[j]
                print(f"l = final_m[{j}] -- {l} = {final_m[j]}")
                print(f"Quindi Theta[{i}][{j}] = Q[k][l] = Q[{k}][{l}] = {Q[k][l]}")
                Theta[i][j] = Q.item((k,l))
     
    print(
        f"-- Theta calcolato matematicamente --\n{np.multiply((((np.transpose(Pprime)).dot(Q)).dot(Pprime)), (A))}")
    print(f"-- Theta calcolato con l'algoritmo --\n{Theta}")
    
    if (Theta == (np.multiply((((np.transpose(Pprime)).dot(Q)).dot(Pprime)), (A)))).all() == True:
        print("Equals")
    else:
        print("Not equals")
        exit(-1)
    
    return Pprime, Theta, m

def inverse(dic, n):
    #print(f"||| Chiamata funzione inverse|||\n{dic} e n = {n}")
    inverted = np.arange((n))
    for i in range(n):
        #print(f"Inverted in posizione dic[{i}] = {dic[i]} assegno {i}")
        inverted[dic[i]] = i
    #print(f"Alla fine inverted = {inverted}")
    return inverted


def complete(m, dic, n):
    filled = np.arange((n))
    for i in range(n):
        if(i in m):
            filled[i] = dic[m[i]]
        else:
            filled[i] = dic[i]
    return filled


def QALS_gless(d_min, eta, i_max, k, lambda_zero, n, N_max, p_delta, q, A, Q):
    m_one = np.arange((n))
    m_two = np.arange((n))
    m = np.arange((n))
    I = np.identity(n)
    P = I
    p = 1
    P_one, Theta_one, m_one = new_g(A.copy(), P.copy(), Q.copy(), m_one.copy(), 1)
    P_two, Theta_two, m_two = new_g(A.copy(), P.copy(), Q.copy(), m_two.copy(), 1)
    # for i in range(k):
    #z_one = (np.transpose(P_one)).dot(random_z(n))
    #z_two = (np.transpose(P_one)).dot(random_z(n))
    z_one = (np.transpose(P_one.copy())).dot(minimization(Theta_one.copy()))
    z_two = (np.transpose(P_two.copy())).dot(minimization(Theta_two.copy()))
    f_one = function_f(Q.copy(), z_one.copy())
    f_two = function_f(Q.copy(), z_two.copy())
    if (f_one < f_two).all():
        z_star = z_one
        f_star = f_one
        P_star = P_one
        m_star = m_one
        z_prime = z_two
    else:
        z_star = z_two
        f_star = f_two
        P_star = P_two
        m_star = m_two
        z_prime = z_one
    if (f_one == f_two).all() == False:
        S = (np.outer(z_prime, z_prime) - I) + np.diagflat(z_prime)
    else:
        S = np.zeros((n, n))
    e = 0
    d = 0
    i = 0
    lam = lambda_zero
    print(f"Entriamo con questo m: {m_star}")
    while True:
        # , end = "\r")
        print(f"-- Ciclo numero {i + 1} con e = {e} e d = {d}")
        Q_prime = np.add(Q, (np.multiply(lam, S)))
        if (i % n == 0):
            p = p - ((p - p_delta)*eta)
        P, Theta_prime, m = new_g(A.copy(), P_star.copy(), Q_prime.copy(), m_star.copy(), p)
        # for i in range(k):
        #z_prime = (np.transpose(P)).dot(random_z(n))
        z_prime = (np.transpose(P.copy())).dot(minimization(Theta_prime.copy()))
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
        if make_decision(q):
            z_prime = h(z_prime.copy(), q)
        if (z_prime == z_star).all() == False:
            f_prime = function_f(Q.copy(), z_prime.copy())
            if (f_prime < f_star).all():
                z_prime, z_star = z_star, z_prime
                f_star = f_prime
                P_star = P
                m_star = m
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
                    m_star = m
                    e = 0
            # R:37 lambda diminuirebbe
            # lam = lam - i/i_max
        else:
            e = e + 1
        i = i + 1
        if (i == i_max):
            print(
                f"-- Raggiunto i max -- \n-- Questo è lo z finora ottenuto --\n{z_star}")
            break
        elif ((e + d >= N_max) and (d < d_min)):
            print(
                f"-- Raggiunta la convergenza al ciclo {i} --\n-- Questo è lo z ottenuto alla convergenza --\n{z_star}")
            break
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
    n = 8

    """
    Solo per test
    """
    rows = 1
    columns = 1

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

    if(n % 8 == 0) and (rows * columns * 8 == n):
        A = dnx.chimera_graph(rows, columns)
        matrix_A = (nx.adjacency_matrix(A)).todense()
    else:
        exit("Error", -1)

    print(
        f"FATTO!\n--------------- A matrice {matrix_A.shape} ---------------\n{matrix_A}\n")
    if(input("Vuoi vedere il grafo di A (S/n)? ") in ["S", "s", "y", "Y"]):
        show_graph(matrix_A)

    print("\n")

    start_time = time.time()
    QALS_gless(d_min, eta, i_max, k, lambda_zero,
               n, N_max, p_delta, q, matrix_A, Q)

    print("\n------------ Impiegati %0.2f secondi con la funzione di g modificata ------------\n" %
          (time.time() - start_time))


if __name__ == "__main__":
    main()
