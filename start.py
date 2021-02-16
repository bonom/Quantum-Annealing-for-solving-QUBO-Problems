#!/usr/bin/env python3

from QA4QUBO import matrix, vector, solver
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
qap = [f for f in listdir("QA4QUBO/tests/") if isfile(join("QA4QUBO/tests/", f))]
MAX = 1000

np.set_printoptions(threshold=sys.maxsize)

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
    return DIR, qap[problem]
"""
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
"""

def write(dir, string):
    file = open(dir, 'a')
    file.write(string+'\n')
    file.close()

def generate_file_npp(_n:int):
    nok = True
    i = 0
    max_range = MAX
    dir = "output_"+str(_n)+"_"+ str(max_range)
    while(nok):
        try:
            with open("output/"+dir+".txt", "r") as file:
                pass
            max_range = int(max_range/2)
            if(not max_range):
                exit("File output terminati")
            dir = "output_"+str(_n)+"_"+ str(max_range)
            i += 1
        except IOError:
            nok = False
        
    DIR = "output/"+dir+".txt"
    open(DIR, 'a').close()

    return DIR, max_range

def file_qap(name):
    nok = True
    i = 0
    dir = "output_"+str(name)+"_"+ str(i)
    while(nok):
        try:
            with open("output/"+dir+".txt", "r") as file:
                pass
            i += 1
            dir = "output_"+str(name)+"_"+ str(i)
        except IOError:
            nok = False
        
    DIR = "output/"+dir+".txt"
    open(DIR, 'a').close()

    return DIR

def main(_n):    
    
    QAP = input("Do you want to use a QAP problem? (y/n) ")
    if(QAP in ['y', 'Y', 1, 's', 'S']):
        QAP = True
        _dir, name = getproblem()
        _Q, penalty, _n, y = matrix.generate_QAP_problem(_dir)
        #with open(_dir) as file:
        #    _Q, _n = matrix.generate_QAP_problem(file)
        name = name.replace(".txt","")
        _DIR = file_qap(name)

    else:
        QAP = False
        if _n == 0:
            _n = int(input("Insert n: "))
        _DIR, max_range = generate_file_npp(_n)
        S = vector.generate_S(_n, max_range)
        _Q, c = matrix.generate_QUBO_problem(S)
    #exit()
    string = " ---------- Problem start ----------\n"
    print(string)
    write(_DIR, string)
    
    if not QAP:
        string = "\n S = "+str(S)+"\n"
        print(string)
        write(_DIR, string)
    
    z = solver.solve(d_min = 70, eta = 0.01, i_max = 8, k = 10, lambda_zero = 1, n = _n, N = 10, N_max = 100, p_delta = 0.1, q = 0.2, topology = 'pegasus', Q = _Q, DIR = _DIR, sim = False)
    
    min_z = solver.function_f(_Q,np.atleast_2d(z).T).item()
    
    string = "So far we found:\n- z - \n"+str(np.atleast_2d(z).T)+"\nand has minimum = "+str(min_z)+"\n"
    if not QAP:
        diff2 = (c**2 + 4*min_z)

        try:
            string += "c = "+str(c)+", c^2 = "+str(c**2)+", diff^2 = "+str(diff2)+", diff = "+str(np.sqrt(diff2))+"\n"
            print(string)
            write(_DIR, string)
        except:
            string += "c = "+str(c)+", c^2 = "+str(c**2)+", diff^2 = "+str(diff2)+"\n"
            print(string)
            write(_DIR, string)
    else:
        string += "y = "+str(y)+", penalty = "+str(penalty)+"\n"
        print(string)
        write(_DIR, string)


if __name__ == '__main__':
    try:
        n = int(sys.argv[1])
    except:
        n = 0
    main(n)