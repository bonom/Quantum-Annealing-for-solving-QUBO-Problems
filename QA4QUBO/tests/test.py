#!/usr/bin/env python3

from QA4QUBO import matrix, vector, solver
import sys

def main():
    #this is only for testing
    #from QA4QUBO.tests.functions import make_decision, shuffle_vector
    
    try:
        from QA4QUBO.functions import make_decision, shuffle_vector
    except:
        raise ImportError

    DIR = input("Please insert the path of file containing data: ")
    
    try:
        with open(DIR) as file:
            _Q = matrix.generate_QAP_problem(file)
            _n = len(_Q)
    except:
        raise FileNotFoundError

    #S = vector.generate_S(_n)
    #_A = matrix.generate_chimera(_n)
    _A = matrix.generate_pegasus(_n)
    #_Q, c = matrix.generate_QUBO_problem(S)
    
    print(" --- Problem start ---")
    try:
        print(f"\n S = {S}\n")
    except:
        pass
    
    #for row in _Q:
    #    print(f"{row}")
    #print(f"\n{_A}\n ---------------------")

    z = solver.solve(d_min = 30, eta = 0.01, i_max = 3000, k = 1, lambda_zero = 1.0, n = _n, N = 8, N_max = 50, p_delta = 0.2, q = 0.1, A = _A, Q = _Q, make_decision = make_decision, shuffle_vector = shuffle_vector)
    min_z = solver.function_f(_Q,z)
    try:
        print(f"So far we found:\n- z - \n{z}\nand has minimum = {min_z}\nc = {c}, c^2 = {c**2}, diff = {c**2 + 4*min_z}")
    except:
        print(f"So far we found:\n- z - \n{z}\nand has minimum = {min_z}\n")



if __name__ == '__main__':
    main()