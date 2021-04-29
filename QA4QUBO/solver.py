#!/usr/bin/env python3
import time
import numpy as np
from QA4QUBO.matrix import generate_chimera, generate_pegasus
from QA4QUBO.script import annealer, hybrid
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridSampler
import datetime
import neal
import sys
import csv
from random import SystemRandom
from QA4QUBO.colors import colors
from dwave.system.composites.embedding import EmbeddingComposite
random = SystemRandom()
np.set_printoptions(linewidth=np.inf,threshold=sys.maxsize)

################################################# USE FOR TESTS MADE
#import sys                                      #
#import random                                   #
#random_seed = random.randint(2**60,2**64)       #
#random.seed(random_seed)                        #
#################################################

def function_f(Q, x):
    return np.matmul(np.matmul(x, Q), np.atleast_2d(x).T)

def make_decision(probability):
    return random.random() < probability

def random_shuffle(a):
    keys = list(a.keys())
    values = list(a.values())
    random.shuffle(values)
    return dict(zip(keys, values))


def shuffle_vector(v):
    n = len(v)
    
    for i in range(n-1, 0, -1):
        j = random.randint(0,i) 
        v[i], v[j] = v[j], v[i]

def shuffle_map(m):
    
    keys = list(m.keys())
    shuffle_vector(keys)
    
    i = 0

    for key, item in m.items():
        it = keys[i]
        ts = item
        m[key] = m[it]
        m[it] = ts
        i += 1

def fill(m, perm, _n):
    n = len(perm)
    if (n != _n):
        n = _n
    filled = np.zeros(n, dtype=int)
    for i in range(n):
        if i in m.keys():
            filled[i] = perm[m[i]]
        else:
            filled[i] = perm[i]

    return filled


def inverse(perm, _n):
    n = len(perm)
    if(n != _n):
        n = _n
    inverted = np.zeros(n, dtype=int)
    for i in range(n):
        inverted[perm[i]] = i

    return inverted


def map_back(z, perm):
    n = len(z)
    inverted = inverse(perm, n)

    z_ret = np.zeros(n, dtype=int)

    for i in range(n):
        z_ret[i] = int(z[inverted[i]])

    return z_ret
     
def g(Q, A, oldperm, p, sim):
    n = len(Q)
    m = dict()
    for i in range(n):
        if make_decision(p):
            m[i] = i
    

    m = random_shuffle(m)
    perm = fill(m, oldperm, n)
    inversed = inverse(perm, n)
    
    Theta = dict()
    if (sim):
        for row, col in A:
            k = inversed[row]
            l = inversed[col]
            Theta[row, col] = Q[k][l]
    else:
        support = dict(zip(A.keys(), np.arange(n))) 
        for key in list(A.keys()):
            k = inversed[support[key]]
            Theta[key, key] = Q[k][k]
            for elem in A[key]:
                l = inversed[support[elem]]
                Theta[key, elem] = Q[k][l]
              
    return Theta, perm

def h(vect, pr):
    n = len(vect)

    for i in range(n):
        if make_decision(pr):
            vect[i] = int((vect[i]+1) % 2)

    return vect

def write(dir, string):
    file = open(dir, 'a')
    file.write(string+'\n')
    file.close()

def get_active(sampler, n):
    nodes = dict()
    tmp = list(sampler.nodelist)
    nodelist = list()
    for i in range(n):
        try:
            nodelist.append(tmp[i])
        except IndexError:
            input(f"Error when reaching {i}-th element of tmp {len(tmp)}") 

    for i in nodelist:
        nodes[i] = list()

    for node_1, node_2 in sampler.edgelist:
        if node_1 in nodelist and node_2 in nodelist:
            nodes[node_1].append(node_2)
            nodes[node_2].append(node_1)

    if(len(nodes) != n):
        i = 1
        while(len(nodes) != n):
            nodes[tmp[n+i]] = list()

    return nodes


def counter(vector):
    count = 0
    for i in range(len(vector)):
        if vector[i]:
            count += 1
    
    return count

def csv_write(DIR, l):
    with open(DIR, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(l)

def now():
    return datetime.datetime.now().strftime("%H:%M:%S")

def solve(d_min, eta, i_max, k, lambda_zero, n, N, N_max, p_delta, q, topology, Q, log_DIR, sim):
    
    try:
        try:
            print(now()+" ["+colors.BOLD+colors.OKGREEN+"LOG"+colors.ENDC+"] Random seed: "+ str(random_seed)+"\n")
        except:
            pass
        if (not sim):
            print(now()+" ["+colors.BOLD+colors.OKGREEN+"LOG"+colors.ENDC+"] "+colors.HEADER+"Started Algorithm in Quantum Mode"+colors.ENDC)
            sampler = DWaveSampler({'topology__type':topology})
            print(now()+" ["+colors.BOLD+colors.OKGREEN+"LOG"+colors.ENDC+"] "+colors.HEADER+"Using Pegasus Topology \n"+colors.ENDC)
            A = get_active(sampler, n)
            sampler = EmbeddingComposite(sampler)
            log_DIR.replace("TSP_","TSP_QA_")
        else:
            print(now()+" ["+colors.BOLD+colors.OKGREEN+"LOG"+colors.ENDC+"] "+colors.OKCYAN+"Started Algorithm in Simulating Mode"+colors.ENDC)
            sampler = neal.SimulatedAnnealingSampler()
            log_DIR.replace("TSP_","TSP_SA_")
            if(topology == 'chimera'):
                print(now()+" ["+colors.BOLD+colors.OKGREEN+"LOG"+colors.ENDC+"] "+colors.OKCYAN+"Using Chimera Topology \n"+colors.ENDC)
                if(n > 2048):
                    n = int(input(
                        now()+" ["+colors.WARNING+colors.BOLD+"WARNING"+colors.ENDC+f"] {n} inserted value is bigger than max topology size (2048), please insert a valid n or press any key to exit: "))
                try:
                    A = generate_chimera(n)
                except:
                    exit()
            else:
                print(now()+" ["+colors.BOLD+colors.OKGREEN+"LOG"+colors.ENDC+"] "+colors.HEADER+"Using Pegasus Topology \n"+colors.ENDC)
                A = generate_pegasus(n)

        print(now()+" ["+colors.BOLD+colors.OKGREEN+"DATA IN"+colors.ENDC+"] dmin = "+str(d_min)+" - eta = "+str(eta)+" - imax = "+str(i_max)+" - k = "+str(k)+" - lambda 0 = "+str(lambda_zero)+" - n = "+str(n) + " - N = "+str(N) + " - Nmax = "+str(N_max)+" - pdelta = "+str(p_delta)+" - q = "+str(q)+"\n")
        
        I = np.identity(n)
        p = 1
        Theta_one, m_one = g(Q, A, np.arange(n), p, sim)
        Theta_two, m_two = g(Q, A, np.arange(n), p, sim)

        print(now()+" ["+colors.BOLD+colors.OKGREEN+"ANN"+colors.ENDC+"] Working on z1...", end=' ')
        start = time.time()
        z_one = map_back(annealer(Theta_one, sampler, k), m_one)
        #z_one = map_back(hybrid(Theta_one, sampler), m_one)
        convert_1 = datetime.timedelta(seconds=(time.time()-start))
        print("Ended in "+str(convert_1)+"\n"+now()+" ["+colors.BOLD+colors.OKGREEN+"ANN"+colors.ENDC+"] Working on z2...", end=' ')
        start = time.time()
        z_two = map_back(annealer(Theta_two, sampler, k), m_two)
        #z_two = map_back(hybrid(Theta_two, sampler), m_two)
        convert_2 = datetime.timedelta(seconds=(time.time()-start))
        print("Ended in "+str(convert_2)+"\n")

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
        
        if (f_one != f_two):
            S = (np.outer(z_prime, z_prime) - I) + np.diagflat(z_prime)
        else:
            S = np.zeros((n,n))
            #S = [[0 for i in range(n)] for j in range(n)] #Old

    except KeyboardInterrupt:
        exit("\n\n["+colors.BOLD+colors.OKGREEN+"KeyboardInterrupt"+colors.ENDC+"] Closing program...")

    e = 0
    d = 0
    i = 1
    lam = lambda_zero
    sum_time = 0
    
    while True:
        print(f"---------------------------------------------------------------------------------------------------------------")
        start_time = time.time()
        if sum_time:
            string = str(datetime.timedelta(seconds=((sum_time/i) * (i_max - i))))
        else:
            string = " not yet available"
        print(now()+" ["+colors.BOLD+colors.OKGREEN+"PRG"+colors.ENDC+f"] Cycle {i}/{i_max} -- {round((((i-1)/i_max)*100), 2)}% -- ETA {string}")

        try:
            Q_prime = np.add(Q, (np.multiply(lam, S)))
            
            if (i % N == 0):
                p = p - ((p - p_delta)*eta)

            Theta_prime, m = g(Q_prime, A, m_star, p, sim)
            
            print(now()+" ["+colors.BOLD+colors.OKGREEN+"ANN"+colors.ENDC+"] Working on z'...", end=' ')
            start = time.time()
            z_prime = map_back(annealer(Theta_prime, sampler, k), m)
            #z_prime = map_back(hybrid(Theta_prime, sampler), m)
            convert_z = datetime.timedelta(seconds=(time.time()-start))
            print("Ended in "+str(convert_z))

            if make_decision(q):
                z_prime = h(z_prime, p)

            if (z_prime != z_star).any() :
                f_prime = function_f(Q, z_prime).item()
                
                if (f_prime < f_star):
                    z_prime, z_star = z_star, z_prime
                    f_star = f_prime
                    m_star = m
                    e = 0
                    d = 0
                    S = S + ((np.outer(z_prime, z_prime) - I) +
                             np.diagflat(z_prime))
                else:
                    d = d + 1
                    if make_decision((p-p_delta)**(f_prime-f_star)):
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
                print(now()+" ["+colors.BOLD+colors.OKGREEN+"DATA"+colors.ENDC+f"] f_prime = {round(f_prime,2)}, f_star = {round(f_star,2)}, p = {p}, e = {e}, d = {d} and lambda = {round(lam,5)}\n"+now()+" ["+colors.BOLD+colors.OKGREEN+"DATA"+colors.ENDC+f"] Took {converted} in total")
                csv_write(DIR=log_DIR,l=[i, f_prime, f_star, p, e, d, lam, z_prime, z_star])
            except UnboundLocalError:
                print(now()+" ["+colors.BOLD+colors.OKGREEN+"DATA"+colors.ENDC+f" No variations on f and z. p = {p}, e = {e}, d = {d} and lambda = {round(lam,5)}\n"+now()+" ["+colors.BOLD+colors.OKGREEN+"DATA"+colors.ENDC+f"] Took {converted} in total")
                csv_write(DIR=log_DIR,l=[i, "null", f_star, p, e, d, lam, "null", z_star])
            
            sum_time = sum_time + (time.time() - start_time)

            print(f"---------------------------------------------------------------------------------------------------------------\n")
            if ((i == i_max) or ((e + d >= N_max) and (d < d_min))):
                if(i != i_max):
                    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+"] Exited at cycle " + str(i)+"/"+str(i_max) + " thanks to convergence.")
                else:
                    print(now()+" ["+colors.BOLD+colors.OKBLUE+"END"+colors.ENDC+"] Exited at cycle "+str(i)+"/"+str(i_max)+"\n")
                break
            
            i = i + 1
        except KeyboardInterrupt:
            break

    converted = datetime.timedelta(seconds=sum_time)
    if i != 1:
        conv = datetime.timedelta(seconds=int(sum_time/(i-1)))
    else:
        conv = datetime.timedelta(seconds=int(sum_time))
    
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"TIME"+colors.ENDC+"] Average time for iteration: " + str(conv)+"\n"+now()+" ["+colors.BOLD+colors.OKGREEN+"TIME"+colors.ENDC+"] Total time: "+str(converted)+"\n")

    return np.atleast_2d(np.atleast_2d(z_star).T).T[0], conv



""" #Old functions

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

    mat = list(zip(rows, cols, values))
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

#there is new p swap
def sim_ann(p, f_prime, f_star):
    if np.log(p) != 0:
        T = -(1/(np.log(p)))
        return np.exp(-(f_prime - f_star)/T)
    return 0

"""