#!/usr/bin/env python3

from QA4QUBO import matrix, vector, solver
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
qap = [f for f in listdir("QA4QUBO/tests/") if isfile(join("QA4QUBO/tests/", f))]
npp = [f for f in listdir("QA4QUBO/npp/") if isfile(join("QA4QUBO/npp/", f))]
MAX = 10000

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
        n = int(file.readline())
        for i in range(n):
            S.append(int(file.readline().rstrip("\n")))

    return S, n


def write(dir, string):
    file = open(dir, 'a')
    file.write(string+'\n')
    file.close()

def main(_n):  
    nok = True
    i = 0
    max_range = 10000
    dir = "output_"+str(_n)+"_"+ str(max_range)
    while(nok):
        try:
            with open("output/"+dir+".txt", "r") as file:
                pass
            max_range = int(MAX/(2**i))
            dir = "output_"+str(_n)+"_"+ str(max_range)
            i += 1
        except IOError:
            nok = False
        
    _DIR = "output/"+dir+".txt"
    open(_DIR, 'a').close()
    
    """
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
    """
    QAP = False
    S = vector.generate_S(_n, max_range)
    _Q, c = matrix.generate_QUBO_problem(S)
    """
    if(input("Do you want to use chimera or pegasus? (c = chimera, p = pegasus) ") in ['C', 'c']):
        print("Will generate a chimera topology")
        _A = matrix.generate_chimera(_n)
    else:
        print("Will generate a pegasus topology")
        _A = matrix.generate_pegasus(_n)
    """
    _A = matrix.generate_pegasus(_n)

    string = " ---------- Problem start ----------\n"
    print(string)
    write(_DIR, string)
    """
    if not QAP:
        string = "\n S = "+str(S)+"\n"
        print(string)
        write(_DIR, string)
    """

    view = False
    """
    if (_n > 16):
        if(input(f"n is very big ({_n}), do you still want to see Q and A? (y/n) ") in ['y', 'Y', 1, 's', 'S']):
            view = True
        else:
            view = False
    """
    if view:
        string = " ---------- Q ---------- \n"
        print(string)
        write(_DIR, string)
        for row in _Q:
            string = str(row)
            print(row)
            write(_DIR, string)
        string = "\n---------- A ----------\n"+str(_A)+"\n -----------------------"
        print(string)
        write(_DIR, string)
    
    z = solver.solve(d_min = 30, eta = 0.01, i_max = 1000, k = 1, lambda_zero = 1.0, n = _n, N = 8, N_max = 50, p_delta = 0.2, q = 0.1, A = _A, Q = _Q, DIR = _DIR)
    min_z = solver.function_f(_Q,z).item()
    string = "So far we found:\n- z - \n"+str(z)+"\nand has minimum = "+str(min_z)+"\n"
    try:
        string += "c = "+str(c)+", c^2 = "+str(c**2)+", diff = "+str(c**2 + 4*min_z)+"\n"
        print(string)
        write(_DIR, string)
    except:
        print(string)
        write(_DIR, string)



if __name__ == '__main__':
    try:
        n = int(sys.argv[1])
    except:
        n = int(input("Insert n: "))
    main(n)