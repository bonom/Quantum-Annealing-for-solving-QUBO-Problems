#!/usr/bin/env python3

from QA4QUBO import matrix, vector, solver
import sys

def main(_n):
    if input("Would you like to test with 'float.txt' and 'integer.txt'? ") in ['1','y','Y']:
        try:
            from QA4QUBO.tests.functions import make_decision, shuffle_vector
        except: 
            raise ImportError
    else:
        try:
            from QA4QUBO.functions import make_decision, shuffle_vector
        except:
            raise ImportError

    if _n == 8:
        S = [25, 7, 13, 31, 42,17, 21,10]
        _A = matrix.generate_chimera(1, 1)
    elif _n == 16:
        S = [1,2,3,4,5,6,7,8,9,8,7,6,6,5,4,5]
        _A = matrix.generate_chimera(1, 2)
    else:
        S = vector.generate_S(_n)
        row = int(input(f"Number of rows for chimera graph? (#nodes = {_n}): "))
        col = int(input(f"Number of columns for chimera graph? (#nodes = {_n}): "))
        _A = matrix.generate_chimera(row,col)

    _Q = matrix.generate_QUBO_problem(S)
    print(" --- Problem start ---")
    print(f"\n S = {S}\n")
    for row in _Q:
        print(f"{row}")
    print(f"\n{_A}\n ---------------------")
    print(solver.solve(d_min = 30, eta = 0.01, i_max = 3000, k = 1, lambda_zero = 1.0, n = _n, N = 8, N_max = 50, p_delta = 0.2, q = 0.1, A = _A, Q = _Q, make_decision = make_decision, shuffle_vector = shuffle_vector))



