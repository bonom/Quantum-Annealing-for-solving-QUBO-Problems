#!/usr/bin/env python3

from QA4QUBO import matrix, vector, solver
import sys

def main(_n):
    #this is only for testing
    #from QA4QUBO.tests.functions import make_decision, shuffle_vector
    
    try:
        from QA4QUBO.functions import make_decision, shuffle_vector
    except:
        raise ImportError

    S = [25, 7, 13, 31, 42,17, 21,10] #S di test
    #S = vector.generate_S(_n)
    _A = matrix.generate_chimera(_n)

    _Q, c = matrix.generate_QUBO_problem(S)
    print(" --- Problem start ---")
    print(f"\n S = {S}\n")
    for row in _Q:
        print(f"{row}")
    print(f"\n{_A}\n ---------------------")
    z = solver.solve(d_min = 30, eta = 0.01, i_max = 3000, k = 1, lambda_zero = 1.0, n = _n, N = 8, N_max = 50, p_delta = 0.2, q = 0.1, A = _A, Q = _Q, make_decision = make_decision, shuffle_vector = shuffle_vector)
    min_z = solver.function_f(_Q,z)
    print(f"So far we found:\n- z - \n{z}\nand has minimum = {min_z}\nc = {c}, c^2 = {c**2}, diff = {c**2 + 4*min_z}")



