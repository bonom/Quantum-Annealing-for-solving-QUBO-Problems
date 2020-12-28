#!/usr/bin/env python3

import time
import random
import numpy as np
from scipy import sparse
from QA4QUBO.script import annealer
from QA4QUBO.hd import run_annealer
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import LeapHybridSampler
import datetime

def function_f(Q, x):
    return ((np.atleast_2d(x).T).dot(Q)).dot(x)

def make_decision(probability):
    return random.random() < probability 

def shuffle_vector(v):
    n = len(v)
    
    for i in range(n-1, 0, -1):
        j = random.randint(0,i) 
        v[i], v[j] = v[j], v[i]

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
        Theta[row][col] = Q[k][l]
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

def write(dir, string):
    file = open(dir, 'a')
    file.write(string+'\n')
    file.close()

def solve(d_min, eta, i_max, k, lambda_zero, n, N, N_max, p_delta, q, A, Q, DIR):
    string = "\n---------- Started Algorithm ----------\n"
    print(string)
    write(DIR, string)
    #sampler = DWaveSampler()
    #sampler = EmbeddingComposite(sampler)
    sampler = LeapHybridSampler()
    I = np.identity(n)
    p = 1
    Theta_one, m_one = g(Q, A, np.arange(n), p)
    Theta_two, m_two = g(Q, A, np.arange(n), p)
    
    start = time.time()
    
    for kindex in range(k):
        #z_one = map_back(annealer(Theta_one, sampler), m_one)
        z_one = annealer(Theta_one, sampler)
        #z_two = map_back(annealer(Theta_two, sampler), m_two)
        z_two = annealer(Theta_two, sampler)
        
    string = "z_one = "+str(np.atleast_2d(z_one).T)+"\nz_two = "+str(np.atleast_2d(z_two).T)+"\n"
    print(string)
    write(DIR, string)
    converted = datetime.timedelta(seconds=((((time.time() - start)/2)/k)*i_max))
    string = "Expected time to complete (determine with i_max): "+str(converted)+"\n"
    print(string)
    write(DIR, string)

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
        try:
            Q_prime = np.add(Q, (np.multiply(lam, S)))
            if (i % N == 0):
                p = p - ((p - p_delta)*eta)

            Theta_prime, m = g(Q_prime, A, m_star, p)

            for kindex in range(k):
                start_quantum = time.time()
                #z_prime = map_back(annealer(Theta_prime, sampler), m)
                time_quantum = time.time()-start_quantum
                start_hybrid = time.time()
                #z_prime_h = map_back(run_annealer(Theta_prime), m)
                z_prime = annealer(Theta_prime, sampler)
                time_hybrid = time.time()- start_hybrid

            if make_decision(q):
                z_prime = h(z_prime, q)

            if (z_prime == z_star) == False:
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
            converted = datetime.timedelta(seconds=(time.time()-start_time))

            try:
                conv_quantum = datetime.timedelta(seconds=time_quantum)
                conv_hybrid = datetime.timedelta(seconds=time_hybrid)
                string = "-- -- Valori ciclo "+str(i)+"/"+str(i_max)+" -- --\np = "+str(p)+", f_prime = "+str(f_prime)+", f_star = "+str(f_star)+", e = "+str(e)+", d = "+str(d)+", Nmax = "+str(N_max)+", dmin = "+str(d_min)+" e lambda = "+str(lam)+"\nz* = "+str(np.atleast_2d(z_star).T)+"\nz' = "+str(np.atleast_2d(z_prime).T)+"\nCi ho messo "+str(converted)+" in totale\n"+"Tempo 'quantum': "+str(conv_quantum)+"\nTempo 'hybrid': "+str(conv_hybrid)+"\nTempo 'calcoli': "+str(converted-(conv_quantum+conv_hybrid))+"\n"
                print(string)
                write(DIR, string)
                #print(f"-- -- Valori ciclo {i}/{i_max} -- --\np = {p}, f_prime = {f_prime}, f_star = {f_star}, e = {e}, d = {d}, Nmax = {N_max}, dmin = {d_min} e lambda = {lam}\nz = {np.atleast_2d(z_star).T}\nCi ho messo {time.time()-start_time} secondi\n")
            except:
                string = "-- -- Valori ciclo "+str(i)+"/"+str(i_max)+" -- --\nNon ci sono variazioni di f, z\ne = "+str(e)+", d = "+str(d)+", Nmax = "+str(N_max)+", dmin = "+str(d_min)+" e lambda = "+str(lam)+"\nz* = "+str(np.atleast_2d(z_star).T)+"\nCi ho messo "+str(converted)+"\n"+"Tempo 'quantum': "+str(conv_quantum)+"\nTempo 'hybrid': "+str(conv_hybrid)+"\nTempo 'calcoli': "+str(converted-(conv_quantum+conv_hybrid))+"\n"
                print(string)
                write(DIR, string)
                #print(f"-- -- Ciclo {i}/{i_max} -- --\n\nNon ci sono variazioni di f, z\ne = {e}, d = {d}, Nmax = {N_max} e dmin = {d_min}\nCi ho messo {time.time()-start_time} secondi\n")
                    
            sum_time = sum_time + (time.time() - start_time)

            if ((i == i_max) or ((e + d >= N_max) and (d < d_min))):
                if(i != i_max):
                    string = "Uscito al ciclo "+str(i)+"/"+str(i_max)+" ed è stata raggiunta la convergenza."
                    print(string)
                    write(DIR, string)
                else:
                    string = "Uscito al ciclo "+str(i)+"/"+str(i_max)+"\n"
                    print(string)
                    write(DIR, string)
                break
            
            i = i + 1
        except KeyboardInterrupt:
            sum_time = sum_time + (time.time() - start_time)
            break

    converted = datetime.timedelta(seconds=sum_time)  
    conv = datetime.timedelta(seconds=(sum_time/i))  
    string = "Tempo medio per iterazione: "+str(conv)+"\nTempo totale: "+str(converted)+"\n"
    print(string)
    write(DIR, string)

    return np.atleast_2d(np.atleast_2d(z_star).T).T