#!/usr/bin/env python3
import time
import numpy as np
from QA4QUBO.matrix import generate_chimera, generate_pegasus
from QA4QUBO.script import annealer
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridSampler
import datetime
import neal
import sys
import csv
from random import SystemRandom
random = SystemRandom()
np.set_printoptions(linewidth=np.inf,threshold=sys.maxsize)
a = list()
rnd = list()

################################################# USE FOR TESTS MADE
#import sys                                      #
#import random                                   #
#random_seed = random.randint(2**60,2**64)       #
#random.seed(random_seed)                        #
#################################################

def function_f(Q, x):
    return np.matmul(np.matmul(x, Q), np.atleast_2d(x).T)

def make_decision(probability):
    tmp = random.random()
    rnd.append(tmp)
    return tmp < probability

def random_shuffle(a):
    keys = list(a.keys())
    values = random.sample(list(a), k=len(a))
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
    filled = [0 for i in range(n)]
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
    inverted = [0 for i in range(n)]
    for i in range(n):
        inverted[perm[i]] = i

    return inverted


def map_back(z, perm):
    n = len(z)
    inverted = inverse(perm, n)

    z_ret = [0 for i in range(n)]

    for i in range(n):
        z_ret[i] = int(z[inverted[i]])

    return z_ret

def printTheta(dictionary):
    tmp = 1
    for i,j in dictionary:
        if i == tmp:
            print("\n")
            tmp+=1
        print(f"({i},{j}) -> {dictionary[i,j]}",end="\t")
        
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

    if [-1] in vect:
        for i in range(n):
            if make_decision(pr):
                vect[i] = -(vect[i])
    else:
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
        nodelist.append(tmp[i])

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

def solve(d_min, eta, i_max, k, lambda_zero, n, N, N_max, p_delta, q, topology, Q, csv_DIR, sim):
    
    with open(csv_DIR, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(["i", "f'", "f*", "p", "e", "d", "lambda", "z'", "z*"])
    #x = minimization(Q)
    #min_f = function_f(Q, np.atleast_2d(x).T)
    #input(f"Computed x = {np.atleast_2d(x).T}\nComputed min_f = {min_f}")
    try:
        try:
            #string = "Random seed: "+ str(random_seed)+"\n"
            print("Random seed: "+ str(random_seed)+"\n")
            #write(txt_DIR,string)
        except:
            pass
        if (not sim):
            #string = "\n---------- Started Algorithm in Quantum Mode ----------\n"
            print("\n---------- Started Algorithm in Quantum Mode ----------\n")
            #write(txt_DIR, string)
            sampler = DWaveSampler()
            A = get_active(sampler, n)
        else:
            #string = "\n---------- Started Algorithm in Simulating Mode ----------"
            print("\n---------- Started Algorithm in Simulating Mode ----------")
            #write(txt_DIR, string)
            sampler = neal.SimulatedAnnealingSampler()

            if(topology == 'chimera'):
                #string = "----------        Using Chimera Topology        ----------\n"
                print("----------        Using Chimera Topology        ----------\n")
                #write(txt_DIR, string)
                if(n > 2048):
                    n = int(input(
                        f"WARNING: {n} inserted value is bigger than max topology size (2048), please insert a valid n or press any key to exit: "))
                try:
                    A = generate_chimera(n)
                except:
                    exit()
            else:
                #string = "----------        Using Pegasus Topology        ----------\n"
                print("----------        Using Pegasus Topology        ----------\n")
                #write(txt_DIR, string)
                A = generate_pegasus(n)
        """
        #########################################################################
        string = "\n---------- Started Algorithm in Hybrid Mode ----------\n"   #
        print(string)                                                           #
        write(txt_DIR, string)                                                      #TO USE FOR HYBRID
        sampler = LeapHybridSampler()                                           #
        A = generate_pegasus(n)                                                 #
        #########################################################################
        """
        #string = "\n --- DATA --- \ndmin = "+str(d_min)+" - eta = "+str(eta)+" - imax = "+str(i_max)+" - k = "+str(k)+" - lambda 0 = "+str(lambda_zero)+" - n = "+str(n) + " - N = "+str(N) + " - Nmax = "+str(N_max)+" - pdelta = "+str(p_delta)+" - q = "+str(q)+"\n"
        print("\n --- DATA --- \ndmin = "+str(d_min)+" - eta = "+str(eta)+" - imax = "+str(i_max)+" - k = "+str(k)+" - lambda 0 = "+str(
            lambda_zero)+" - n = "+str(n) + " - N = "+str(N) + " - Nmax = "+str(N_max)+" - pdelta = "+str(p_delta)+" - q = "+str(q)+"\n")
        #write(txt_DIR, string)

        #if (input("Data correct?") not in ['1','Y','y']):
        #    exit("Exit...")

        #txt_DIR = txt_DIR+"_matrix.txt"
        #file = open(txt_DIR, 'a')
        #i = 0
        #for row in Q:
        #    i += 1
        #    print(
        #        f"--- Printing Q in file './{dir}' ... {round((i/n)*100, 2)} %", end='\r')
        #    file.write(str(row)+'\n')
#
        #print(f"--- Printing Q in file './{dir}' END ---  ")
        #file.close()
#
        I = np.identity(n)
        p = 1
        Theta_one, m_one = g(Q, A, np.arange(n), p, sim)
        Theta_two, m_two = g(Q, A, np.arange(n), p, sim)

        #string = "Working on z1..."
        print("Working on z1...", end=' ')
        #write(txt_DIR, string)
        start = time.time()
        z_one = map_back(annealer(Theta_one, sampler, k), m_one)
        convert_1 = datetime.timedelta(seconds=(time.time()-start))
        #string = "Ended in "+str(convert_1)+" .\nWorking on z2..."
        print("Ended in "+str(convert_1)+" .\nWorking on z2...", end=' ')
        #write(txt_DIR, string)
        start = time.time()
        z_two = map_back(annealer(Theta_two, sampler, k), m_two)
        convert_2 = datetime.timedelta(seconds=(time.time()-start))
        #string = "Ended in "+str(convert_2)+" ."
        print("Ended in "+str(convert_2)+" .")
        #write(txt_DIR, string)

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
        #string = "KeyboardInterrupt occurred before cycle, closing program..."
        print("KeyboardInterrupt occurred before cycle, closing program...")
        #write(txt_DIR, string)
        exit()

    e = 0
    d = 0
    i = 1
    lam = lambda_zero
    min_so_far = 0

    sum_time = 0
    m = 0
    while True:
        start_time = time.time()
        try:
            string = str(round(((i/i_max)*100), 2))+"% -- ETA: "+str(datetime.timedelta(seconds=((sum_time/i) * (i_max - i))))+"\n"
        except:
            string = str(round(((i/i_max)*100), 2)) + "% -- ETA: not yet available\n"
        print(string)

        try:
            Q_prime = np.add(Q, (np.multiply(lam, S)))
            #dir = DIR+"_Q.txt"
            #file = open(dir, 'w')
            #file.write(str(Q_prime)+"\n")
            #file.close()
            
            if (i % N == 0):
                p = p - ((p - p_delta)*eta)

            Theta_prime, m = g(Q_prime, A, m_star, p, sim)
            
            #dir = DIR+"_m.txt"
            #file = open(dir, 'a')
            #file.write(str(m)+"\n")
            #file.close()
            
            #string = "Working on z'..."
            print("Working on z'...", end=' ')
            #write(txt_DIR, string)
            start = time.time()
            z_prime = map_back(annealer(Theta_prime, sampler, k), m)
            convert_z = datetime.timedelta(seconds=(time.time()-start))
            #string = "Ended in "+str(convert_z)+" ."
            print("Ended in "+str(convert_z)+" .")
            #write(txt_DIR, string)

            if make_decision(q):
                z_prime = h(z_prime, p)

            #mat = np.zeros((n,n))
            #print(Theta_prime)
            #for key, value in Theta_prime.items():
            #    a,b = key
            #    mat[a][b] = value
            #with open('errors.txt', 'a') as f:
            #    f.write("\n"+str(i)+". Vettore:"+str(counter(z_prime))+"\n"+str(z_prime)+"\nMatrice:\n")
            #    f.write(str(mat)+"\n")
            #input("...")

            if (z_prime != z_star) :
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

            #string = "Esco al ciclo "+str(i)+" con questi valori:\n"
            print("Esco al ciclo "+str(i)+" con questi valori:\n")
            #write(txt_DIR,string)
            try:
                if(n > 16):
                    string = "-- -- Values cycle "+str(i)+"/"+str(i_max)+" -- --\np = "+str(p)+", f_prime = "+str(f_prime)+", f_star = "+str(f_star)+", e = "+str(
                        e)+", d = "+str(d)+", Nmax = "+str(N_max)+", dmin = "+str(d_min)+" and lambda = "+str(lam)+"\nTook "+str(converted)+" in total\n"
                else:
                    string = "-- -- Values cycle "+str(i)+"/"+str(i_max)+" -- --\np = "+str(p)+", f_prime = "+str(f_prime)+", f_star = "+str(f_star)+", e = "+str(e)+", d = "+str(
                        d)+", Nmax = "+str(N_max)+", dmin = "+str(d_min)+" and lambda = "+str(lam)+"\nz* = "+str(z_star)+"\nz' = "+str(z_prime)+"\nTook "+str(converted)+" in total\n"
                
                #write(txt_DIR, string)
            except:
                if(n > 16):
                    string = "-- -- Values cycle "+str(i)+"/"+str(i_max)+" -- --\nNo variations on f, z\ne = "+str(e)+", d = "+str(
                        d)+", Nmax = "+str(N_max)+", dmin = "+str(d_min)+" and lambda = "+str(lam)+"\nTook "+str(converted)+" in total\n"
                else:
                    string = "-- -- Values cycle "+str(i)+"/"+str(i_max)+" -- --\nNo variations on f, z\ne = "+str(e)+", d = "+str(d)+", Nmax = "+str(
                        N_max)+", dmin = "+str(d_min)+" and lambda = "+str(lam)+"\nz* = "+str(z_star)+"\nTook "+str(converted)+" in total\n"
                #write(txt_DIR, string)

            print(string)
            with open(csv_DIR, 'a') as file:
                writer = csv.writer(file)
                try:
                    writer.writerow([i, f_prime, f_star, p, e, d, lam, z_prime, z_star])
                except UnboundLocalError:
                    writer.writerow([i, "null", f_star, p, e, d, lam, "null", z_star])
            #dir = DIR+"_vector.txt"
            #file = open(dir, 'a')
            #file.write("Cycle "+str(i)+"-th - z* --> "+str(z_star)+"\n")
            #file.close()
            sum_time = sum_time + (time.time() - start_time)

            if ((i == i_max) or ((e + d >= N_max) and (d < d_min))):
                if(i != i_max):
                    #string = "Exited at cycle " + str(i)+"/"+str(i_max) + " thanks to convergence."
                    print("Exited at cycle " + str(i)+"/"+str(i_max) + " thanks to convergence.")
                    #write(txt_DIR, string)
                else:
                    #string = "Exited at cycle "+str(i)+"/"+str(i_max)+"\n"
                    print("Exited at cycle "+str(i)+"/"+str(i_max)+"\n")
                    #write(txt_DIR, string)
                break
            
            if f_star < min_so_far:
                min_so_far = f_star

            i = i + 1
        except KeyboardInterrupt:
            break

    converted = datetime.timedelta(seconds=sum_time)
    if i != 1:
        conv = datetime.timedelta(seconds=(sum_time/(i-1)))
    else:
        conv = datetime.timedelta(seconds=(sum_time))
    #string = "Average time for iteration: " + str(conv)+"\nTotal time: "+str(converted)+"\n"
    print("Average time for iteration: " + str(conv)+"\nTotal time: "+str(converted)+"\n")
    #write(txt_DIR, string)


    #string = "Min so far: "+str(min_so_far)+"\n"
    print("Min so far: "+str(min_so_far)+"\n")
    #write(txt_DIR, string)

    return np.atleast_2d(z_star).T



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