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
import itertools
import time
import os
import csv
import numpy as np
import sys
from datetime import datetime, timedelta
from QA4QUBO.colors import colors
from QA4QUBO.script import annealer, hybrid
#from colors import colors
from dwave.system.samplers import DWaveSampler           
from dwave.system.composites import EmbeddingComposite   
import neal
from dwave.system import LeapHybridSampler
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

def solve_tsp_brute_force(nodes_array):
    number_of_nodes = len(nodes_array)
    initial_order = range(0, number_of_nodes)
    all_permutations = [list(x) for x in itertools.permutations(initial_order)]
    cost_matrix = get_tsp_matrix(nodes_array)
    best_permutation = all_permutations[0]
    best_cost = calculate_cost(cost_matrix, all_permutations[0])
    for permutation in all_permutations:
        current_cost = calculate_cost(cost_matrix, permutation)
        if current_cost < best_cost:
            best_permutation = permutation
            best_cost = current_cost
    
    return np.array(best_permutation), round(best_cost, 2)

def solve_tsp(qubo_dict, k, tsp_matrix):
    #response = annealer(qubo_dict, neal.SimulatedAnnealingSampler(), k)      
    response = annealer(qubo_dict, EmbeddingComposite(DWaveSampler()), k)            
    return np.array(response)

def hybrid_tsp(qubo_dict, tsp_matrix):   
    response = hybrid(qubo_dict, LeapHybridSampler())          
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
        solution = list()
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
                    if it not in keep: 
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
            if it not in keep: 
                diff.append(it)
            
        for i in range(n):
            if solution[i] == -1 and len(diff) != 0:
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
    csv_write(DIR, l=["nodes", "tsp", "qubo"])
       
    qubo = dict()
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"LOG"+colors.ENDC+"] Creating nodes array ... ")

    if n == 10:
        nodes_array = np.array([[0.66083966, 9.36939755],[1.98485729, 7.62491491],[1.54206421, 1.18410071],[2.01555644, 3.15713817],[7.83888128, 8.77009394],[1.4779611, 4.16581664],[0.6508892, 6.31063212],[6.6267559, 5.45120931],[9.73821452, 2.20299234],[3.50140032, 5.36660266]])
    else:
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
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"LOG"+colors.ENDC+"] Solving bruteforce ... ")

    csv_write(DIR, l=[nodes_array, tsp_matrix, qubo])
    csv_write(DIR, l=[])

    start = time.time()
    bf_solution, bf_cost = solve_tsp_brute_force(nodes_array)
    bf_time = time.time() - start
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+f"] Solved in {timedelta(seconds = int(time.time()-start))}")

    csv_write(DIR, l=["Solution BF", bf_solution])
    csv_write(DIR, l=["Cost BF", bf_cost])
    csv_write(DIR, l=["Time BF", bf_time])
    csv_write(DIR, l=[])

    _start = time.time()
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"LOG"+colors.ENDC+"] Start computing response ... ")
    start = time.time()
    response_QA = solve_tsp(qubo,1000,tsp_matrix)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+f"] Response computed in {timedelta(seconds = int(time.time()-start))}")
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"LOG"+colors.ENDC+"] Start computing solution ... ")
    start = time.time()
    solution_QA = decode_solution(response_QA, True)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+f"] Solution computed in {timedelta(seconds = int(time.time()-start))}")
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"END"+colors.ENDC+"] Computing cost ... ")
    cost_QA = round(calculate_cost(tsp_matrix,solution_QA),2)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+f"] Cost computed ")
    print("\t\t\t"+colors.BOLD+colors.HEADER+" END"+colors.ENDC)
    tme_QA = time.time() - _start

    csv_write(DIR, l=["Response QA", response_QA])
    csv_write(DIR, l=["Solution QA", solution_QA])
    csv_write(DIR, l=["Cost QA", cost_QA])
    csv_write(DIR, l=["Time QA", tme_QA])
    csv_write(DIR, l=[])

    #HYBRID
    _start = time.time()
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"LOG"+colors.ENDC+"] Start computing hybrid response ... ")
    start = time.time()
    response_HY = solve_tsp(qubo,1000,tsp_matrix)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+f"] Hybrid response computed in {timedelta(seconds = int(time.time()-start))}")
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"LOG"+colors.ENDC+"] Start computing hybrid solution ... ")
    start = time.time()
    solution_HY = decode_solution(response_HY, True)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+f"] Hybrid olution computed in {timedelta(seconds = int(time.time()-start))}")
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"END"+colors.ENDC+"] Computing hybrid cost ... ")
    cost_HY = round(calculate_cost(tsp_matrix,solution_HY),2)
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"END"+colors.ENDC+f"] Hybrid cost computed ")
    print("\t\t\t"+colors.BOLD+colors.HEADER+" END"+colors.ENDC)
    tme_HY = time.time() - _start
    
    csv_write(DIR, l=["Response HY", response_HY])
    csv_write(DIR, l=["Solution HY", solution_HY])
    csv_write(DIR, l=["Cost HY", cost_HY])
    csv_write(DIR, l=["Time HY", tme_HY])
    csv_write(DIR, l=[])
    
    return nodes_array, tsp_matrix, qubo#, response_QA, solution_QA, cost_QA, tme_QA, bf_solution, bf_cost, bf_time


if __name__ == '__main__':
    
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

