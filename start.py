#!/usr/bin/env python3

from QA4QUBO import matrix, vector, solver
from os import listdir
from os.path import isfile, join
qap = [f for f in listdir("QA4QUBO/tests/") if isfile(join("QA4QUBO/tests/", f))]
npp = [f for f in listdir("QA4QUBO/npp/") if isfile(join("QA4QUBO/npp/", f))]

def getproblem():
    elements = list()
    i = 0
    for element in qap:
        elements.append(element)
        element = element[:-4]
        print(f"Write {i} for the problem {element}")
        i += 1
    
    problem = int(input("Which problem do you want to select? "))
    DIR = "QA4QUBO/tests/"+qap[problem]
    return DIR

def getS():
    elements = list()
    i = 0
    for element in npp:
        elements.append(element)
        element = element[:-4]
        print(f"Write {i} for the problem {element}")
        i += 1
    
    problem = int(input("Which problem do you want to select? "))
    DIR = "QA4QUBO/npp/"+npp[problem]
    return DIR

def generateS():
    S = list()
    DIR = getS()
    with open(DIR) as file:
        n = int(file.read())
        for i in range(n):
            S.append(file.read())

    print(f" N = {n}\n{S}")
    input()
    return S, n

def main():
    
    QAP = input("Do you want to use a QAP problem? (y/n) ")
    if(QAP in ['y', 'Y', 1, 's', 'S']):
        QAP = True
        DIR = getproblem()
        with open(DIR) as file:
            _Q = matrix.generate_QAP_problem(file)
            _n = len(_Q)
    else:
        QAP = False
        S, _n = generateS()
        _Q, c = matrix.generate_QUBO_problem(S)
    
    
    if(input("Do you want to use chimera or pegasus? (c = chimera, p = pegasus) ") in ['C', 'c']):
        print("Will generate a chimera topology")
        _A = matrix.generate_chimera(_n)
    else:
        print("Will generate a pegasus topology")
        _A = matrix.generate_pegasus(_n)
    
    
    print(" --- Problem start ---")
    if not QAP:
        print(f"\n S = {S}\n")
    
    view = True
    if (_n > 16):
        if(input(f"n is very big ({_n}), do you still want to see Q and A? (y/n) ") in ['y', 'Y', 1, 's', 'S']):
            view = True
        else:
            view = False

    if view:
        for row in _Q:
            print(f"{row}")
        print(f"\n{_A}\n ---------------------")

    z = solver.solve(d_min = 30, eta = 0.01, i_max = 3000, k = 1, lambda_zero = 1.0, n = _n, N = 8, N_max = 50, p_delta = 0.2, q = 0.1, A = _A, Q = _Q)
    min_z = solver.function_f(_Q,z)
    try:
        print(f"So far we found:\n- z - \n{z}\nand has minimum = {min_z}\nc = {c}, c^2 = {c**2}, diff = {c**2 + 4*min_z}")
    except:
        print(f"So far we found:\n- z - \n{z}\nand has minimum = {min_z}\n")



if __name__ == '__main__':
    main()