#!/usr/local/bin/python3
#
#
#
#   THE CODE IN THIS PYTHON SCRIPT IS NOT WRITTEN BY ME
#       I ARRANGED AN ALREADY EXISTING SCRIPT FROM:
#      https://github.com/BOHRTECHNOLOGY/quantum_tsp
#
#
#

import time
import os
import csv
import numpy as np
import sys
from datetime import datetime, timedelta
#from QA4QUBO.colors import colors
from colors import colors
from dwave.system.samplers import DWaveSampler           
from dwave.system.composites import EmbeddingComposite   
import neal
from random import SystemRandom
random = SystemRandom()

def add_cost_objective(distance_matrix, cost_constant, qubo_dict):
    n = len(distance_matrix)
    for t in range(n):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                qubit_a = t * n + i
                qubit_b = (t + 1)%n * n + j
                qubo_dict[(qubit_a, qubit_b)] = cost_constant * distance_matrix[i][j]

def add_time_constraints(distance_matrix, constraint_constant, qubo_dict):
    n = len(distance_matrix)
    for t in range(n):
        for i in range(n):
            qubit_a = t * n + i
            if (qubit_a, qubit_a) not in qubo_dict.keys():
                qubo_dict[(qubit_a, qubit_a)] = -constraint_constant
            else:
                qubo_dict[(qubit_a, qubit_a)] += -constraint_constant
            for j in range(n):
                qubit_b = t * n + j
                if i!=j:
                    qubo_dict[(qubit_a, qubit_b)] = 2 * constraint_constant

def add_position_constraints(distance_matrix, constraint_constant, qubo_dict):
    n = len(distance_matrix)
    for i in range(n):
        for t1 in range(n):
            qubit_a = t1 * n + i
            if (qubit_a, qubit_a) not in qubo_dict.keys():
                qubo_dict[(qubit_a, qubit_a)] = -constraint_constant
            else:
                qubo_dict[(qubit_a, qubit_a)] += -constraint_constant
            for t2 in range(n):
                qubit_b = t2 * n + i
                if t1!=t2:
                    qubo_dict[(qubit_a, qubit_b)] = 2 * constraint_constant

def solve_tsp(qubo_dict, k, tsp_matrix):
    response = list(neal.SimulatedAnnealingSampler().sample_qubo(qubo_dict, num_reads=k).first.sample.values())            
    #response = list(EmbeddingComposite(DWaveSampler()).sample_qubo(qubo_dict, chain_strength=800, num_reads=k).first.sample.values())            
    return np.array(response)

def advance(iter, rnd):
    iterator = next(iter)
    while random.random() > rnd:
        iterator = next(iter, iterator)
    return iterator

def decode_solution(response, validate):
    
    n = int(np.sqrt(len(response)))
    solution = np.array(n)
    raw = dict()
    for i in range(n):
        raw[i] = list()
    keep = list()
    all_ = list()
    diff = list()
    indexes = list()

    if not validate:
        for i in range(n):
            for j in range(n):
                if(response[n*i + j] == 1):
                    last = j
            if (last != -1):
                solution.append(last)
            last = -1
    else:
        solution = np.array([-1 for i in range(n)])
        for i in range(n):
            for j in range(n):
                if (response[n*i +j] == 1):
                    raw[i].append(j)
        
        for i in range(n):
            if len(raw[i]) == 1:
                keep.append(raw[i][0])
                solution[i] = raw[i][0]
            all_.append(i)            

        for i in range(n):
            if len(raw[i]) > 1:
                for it in raw[i]:
                    if it in keep and it == keep[-1]: #OR keep.index(it)???
                        diff.append(it)
                
                if len(diff) > 0:
                    it = advance(iter(diff), random.random() % len(diff))
                    solution[i] = it
                    keep.append(it)
                    diff.clear()

        for i in range(n):
            for j in range(n):
                if solution[j] == i:
                    indexes.append(j) 

            if len(indexes) > 1:
                random.shuffle(indexes)
                index = indexes[0]
                for it in indexes:
                    if it == index: 
                        solution[it] = i 
                    else:
                        solution[it] = -1

                keep.append(i)

            indexes.clear()

        for it in all_:
            if it in keep and it == keep[-1]: #OR keep.index(it)???
                diff.append(it)

        for i in range(n):
            if -solution[i] and len(diff) != 0:
                it = advance(iter(diff), random.random() % len(diff))
                solution[i] = it
                diff.remove(it)

    return solution

def calculate_cost(cost_matrix, solution):
    cost = 0
    for i in range(len(solution)):
        a = i%len(solution)
        b = (i+1)%len(solution)
        cost += cost_matrix[solution[a]][solution[b]]

    return cost

def binary_state_to_points_order(binary_state):
    points_order = []
    number_of_points = int(np.sqrt(len(binary_state)))
    for p in range(number_of_points):
        for j in range(number_of_points):
            if binary_state[(number_of_points) * p + j] == 1:
                points_order.append(j)
    return points_order

def create_nodes_array(N):
    nodes_list = []
    for i in range(N):
        nodes_list.append(np.random.rand(2) * 10)
    return np.array(nodes_list)


def get_tsp_matrix(nodes_array):
    n = len(nodes_array)
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i, n):
            matrix[i][j] = distance(nodes_array[i], nodes_array[j])
            matrix[j][i] = matrix[i][j]
    return matrix


def distance(point_A, point_B):
    return np.sqrt((point_A[0] - point_B[0])**2 + (point_A[1] - point_B[1])**2)

def csv_write(DIR, l):
    with open(DIR, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(l)

def now():
    return datetime.now().strftime("%H:%M:%S")

def tsp(n, DIR):
    print("\t"+colors.BOLD+colors.HEADER+"    CLASSIC TSP PROBLEM SOLVER..."+colors.ENDC)
    
    qubo = dict()
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"LOG"+colors.ENDC+"] Creating nodes array ... ")
    nodes_array = create_nodes_array(n)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+"] Nodes array created")
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"LOG"+colors.ENDC+"] Creating tsp matrix ... ")
    tsp_matrix = get_tsp_matrix(nodes_array)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+"] Tsp matrix created")

    constraint_constant = tsp_matrix.max()*len(tsp_matrix) 
    cost_constant = 1    
          
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"LOG"+colors.ENDC+"] Adding constraints ... ")
    add_cost_objective(tsp_matrix,cost_constant,qubo)
    add_time_constraints(tsp_matrix,constraint_constant,qubo)
    add_position_constraints(tsp_matrix,constraint_constant,qubo)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+"] Contraints added")
    _start = time.time()
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"LOG"+colors.ENDC+"] Start computing response ... ")
    start = time.time()
    response = solve_tsp(qubo,1000,tsp_matrix)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+f"] Response computed in {timedelta(seconds = int(time.time()-start))}")
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"LOG"+colors.ENDC+"] Start computing solution ... ")
    start = time.time()
    solution = decode_solution(response, True)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+f"] Solution computed in {timedelta(seconds = int(time.time()-start))}")
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"END"+colors.ENDC+"] Computing cost ... ")
    cost = round(calculate_cost(tsp_matrix,solution),2)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+f"] Cost computed ")
    print("\t\t\t"+colors.BOLD+colors.HEADER+" END"+colors.ENDC)
    tme = time.time() - _start
    
    csv_write(DIR, l=["nodes", "response", "solution", "cost", "tsp", "qubo"])
    csv_write(DIR, l=[nodes_array, response, solution, cost , tsp_matrix, qubo])

    return nodes_array, tsp_matrix, qubo, response, solution, cost, tme

if __name__ == '__main__':
    #decode_solution([1,0,1,0,0,1,0,1,0], True)

    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        n = sys.argv[1]
    except IndexError:
        n = input("Insert n (3 <= n <= 10): ")

    if (os.getcwd() == "/workspace/Quantum-Annealing-for-solving-QUBO-Problems"):
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        _DIR = "outputs/TSP_"+n+"_"+now()+".csv"
    else:
        if not os.path.exists('../outputs'):
            os.mkdir('../outputs')
        _DIR = "../outputs/TSP_"+n+"_"+now()+".csv"
        
    tsp(int(n), DIR = _DIR)

