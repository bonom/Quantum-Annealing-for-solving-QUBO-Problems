#!/usr/local/bin/python3
from QA4QUBO import matrix, vector, solver
from QA4QUBO.colors import colors
from os import listdir, mkdir, system, name
from os.path import isfile, join, exists
import sys
import numpy as np
import csv
qap = [f for f in listdir("QA4QUBO/qap/") if isfile(join("QA4QUBO/qap/", f))]
#MAX = 1000000000 #1 miliardo
#MAX = 1000000 #1milione
#MAX = 100000 #100k
#MAX = 10000 #10k
#MAX = 1000 #mille
QAP = False
NPP = False

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

def write(dir, string):
    file = open(dir, 'a')
    file.write(string+'\n')
    file.close()

def csv_write(DIR, l):
    with open(DIR, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(l)

def generate_file_npp(_n:int):
    nok = True
    i = 0
    max_range = 1000000
    dir = "NPP_output_"+str(_n)+"_"+ str(max_range)
    while(nok):
        try:
            with open("outputs/"+dir+".csv", "r") as file:
                pass
            max_range = int(max_range/10)
            if(max_range < 10):
                exit("File output terminati")
            dir = "NPP_output_"+str(_n)+"_"+ str(max_range)
            i += 1
        except IOError:
            nok = False
        
    DIR = "outputs/"+dir

    return DIR, max_range

def generate_file_tsp(n:int,edges:int):
    nok = True
    i = 0
    dir = "TSP_output_"+str(n)+"_"+ str(edges)
    while(nok):
        try:
            with open("outputs/"+dir+".csv", "r") as file:
                pass
            i += 1
            dir = "TSP_output_"+str(n)+"_"+ str(edges)+"_"+str(i)
        except IOError:
            nok = False
        
    DIR = "outputs/"+dir

    return DIR

def generate_file_qap(name):
    nok = True
    i = 0
    dir = "QAP_output_"+str(name)+"_"+ str(i)
    while(nok):
        try:
            with open("outputs/"+dir+".csv", "r") as file:
                pass
            i += 1
            dir = "QAP_output_"+str(name)+"_"+ str(i)
        except IOError:
            nok = False
        
    DIR = "outputs/"+dir

    return DIR

def main(_n):    
    print(" --------- Creating Problem ---------\n")
    
    if QAP:
        _dir, name = getproblem()
        _Q, penalty, _n, y = matrix.generate_QAP_problem(_dir)
        name = name.replace(".txt","")
        _DIR = generate_file_qap(name)
        csv_DIR = _DIR.replace("QAP","QAP_LOG") + ".csv"

    elif NPP:
        while _n <= 0:
            _n = int(input("["+colors.FAIL+colors.BOLD+"Invalid n"+colors.ENDC+"] Insert n: "))
        _DIR, max_range = generate_file_npp(_n)
        S = vector.generate_S(_n, max_range)
        _Q, c = matrix.generate_QUBO_problem(S)
        csv_DIR = _DIR.replace("NPP","NPP_LOG") + ".csv"
    
    else:
        while _n <= 0 or _n > 9:
            _n = int(input("["+colors.FAIL+colors.BOLD+"Invalid n"+colors.ENDC+"] Insert n: "))
        G, _Q = matrix.tsp(_n)
        len_edges = len(G.edges())
        _DIR = generate_file_tsp(_n,len_edges)
        csv_DIR = _DIR.replace("TSP","TSP_LOG") + ".csv"

    print(" ---------- Problem start ----------\n")
    
    if NPP:
        print("\n S = "+str(S)+"\n")

    z = np.atleast_2d(solver.solve(d_min = 70, eta = 0.01, i_max = 3, k = 1000, lambda_zero = 3/2, n = _n if QAP or NPP else _n**2, N = 10, N_max = 100, p_delta = 0.1, q = 0.2, topology = 'pegasus', Q = _Q, csv_DIR = csv_DIR, sim = False)).T[0]
    
    min_z = solver.function_f(_Q,z).item()
    
    string = "So far we found:\n- z - \n"+str(z)+"\nand has minimum = "+str(min_z)+"\n"
    if NPP:
        diff2 = (c**2 + 4*min_z)
        string += "c = "+str(c)+", c^2 = "+str(c**2)+", diff^2 = "+str(diff2)+", diff = "+str(np.sqrt(diff2))+"\n"
        print(string)
        csv_write(DIR=_DIR+"_solution.csv", l=["c","c**2","diff**2","diff","S", "z", "Q"])
        csv_write(DIR=_DIR+"_solution.csv", l=[c,c**2,diff2,np.sqrt(diff2),S,z, _Q  if _n < 5 else "too big"])
        
    elif QAP:
        csv_write(DIR=_DIR+"_solution.csv", l=["problem","y","penalty","difference (y+minimum)", "z", "Q" ])
        csv_write(DIR=_DIR+"_solution.csv", l=[name,y,penalty,y+min_z,np.atleast_2d(z).T,_Q])
        string += "y = "+str(y)+", penalty = "+str(penalty)+", difference (y+minimum) = "+str(y+min_z)+"\n"
        print(string)

    else:
        res = np.split(z,_n)
        valid = True
        route = list()
        for split in res:
            if np.count_nonzero(split == 1) != 1:
                valid = False
            route.append(str(np.where(split == 1)))
        if not valid:
            string += "\nIl risultato non è divisibile"
        
        csv_write(DIR=_DIR+"_solution.csv", l=["Result","Route", "z", "res"])
        csv_write(DIR=_DIR+"_solution.csv", l=["Valid" if valid else "Not valid",route, z, res])
        csv_write(DIR=_DIR+"_solution.csv", l=[])
        csv_write(DIR=_DIR+"_solution.csv", l=[])
        csv_write(DIR=_DIR+"_solution.csv", l=["G.Nodes", "G.edges"])
        csv_write(DIR=_DIR+"_solution.csv", l=[G.nodes(), G.edges()])
        print(string)


if __name__ == '__main__':
    system('cls' if name == 'nt' else 'clear')

    if not exists('outputs'):
        mkdir('outputs')

    try:
        n = int(sys.argv[1])
    except IndexError:
        n = 0
    main(n)