#!/usr/local/bin/python3
import pandas as pd
import datetime
import time
from QA4QUBO import matrix, vector, solver, tsp#, solver2
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

def log_write(tpe, var):
    return "["+colors.BOLD+str(tpe)+colors.ENDC+"]\t"+str(var)+"\n"

def getproblem():
    elements = list()
    i = 0
    for element in qap:
        elements.append(element)
        element = element[:-4]
        print(f"Write {i} for the problem {element}")
        i += 1
    
    problem = int(input("Which problem do you want to select? "))
    DIR = "QA4QUBO/qap/"+qap[problem]
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
    _dir = "NPP_"+str(_n)+"_"+ str(max_range)
    while(nok):
        try:
            with open("outputs/"+_dir.replace("NPP","NPP_LOG")+".csv", "r") as file:
                pass
            max_range = int(max_range/10)
            if(max_range < 10):
                exit("File output terminati")
            _dir = "NPP_"+str(_n)+"_"+ str(max_range)
            i += 1
        except FileNotFoundError:
            nok = False
        
    DIR = "outputs/"+_dir

    return DIR, max_range

def generate_file_tsp(n:int):
    nok = True
    i = 1
    _dir = "TSP_"+str(n)+"_"+str(i)
    while(nok):
        try:
            with open("outputs/"+_dir.replace("TSP","TSP_LOG")+".csv", "r") as file:
                pass
            i += 1
            _dir = "TSP_"+str(n)+"_"+str(i)
        except FileNotFoundError:
            nok = False
        
    DIR = "outputs/"+_dir

    return DIR

def generate_file_qap(name):
    nok = True
    i = 0
    _dir = "QAP_"+str(name)
    while(nok):
        try:
            with open("outputs/"+_dir+".csv", "r") as file:
                pass
            i += 1
            _dir = "QAP_"+str(name)+"_"+ str(i)
        except FileNotFoundError:
            nok = False
        
    DIR = "outputs/"+_dir

    return DIR

def convert_qubo_to_Q(qubo, n):
    Q = np.zeros((n,n))
    for x,y in qubo.keys():
        Q[x][y] = qubo[x,y]

    return Q

def main(nn):    
    print("\t\t"+colors.BOLD+colors.WARNING+"  BUILDING PROBLEM..."+colors.ENDC)
    if QAP:
        _dir, name = getproblem()
        _Q, penalty, nn, y = matrix.generate_QAP_problem(_dir)
        name = name.replace(".txt","")
        _DIR = generate_file_qap(name)
        log_DIR = _DIR.replace("QAP","QAP_LOG") + ".csv"

    elif NPP:
        while nn <= 0:
            nn = int(input("["+colors.FAIL+colors.BOLD+"Invalid n"+colors.ENDC+"] Insert n: "))
        _DIR, max_range = generate_file_npp(nn)
        S = vector.generate_S(nn, max_range)
        _Q, c = matrix.generate_QUBO_problem(S)
        log_DIR = _DIR.replace("NPP","NPP_LOG") + ".csv"
    
    else:
        while nn <= 0 or nn > 12:
            nn = int(input("["+colors.FAIL+colors.BOLD+"Invalid n"+colors.ENDC+"] Insert n: "))
        _DIR = generate_file_tsp(nn)
        log_DIR = _DIR.replace("TSP","TSP_LOG") + ".csv"
        csv_write(DIR=log_DIR, l=["i", "f'", "f*", "p", "e", "d", "lambda", "z'", "z*"])
        df = pd.DataFrame(columns=["Solution", "Cost", "Fixed solution", "Fixed cost", "Response time", "Total time", "Response"], index=['Bruteforce', 'D-Wave', 'Hybrid', 'QALS'])
        tsp_matrix, qubo = tsp.tsp(nn, _DIR + "_solution.csv" , _DIR[:-1]+"DATA.csv", df) #, response, solution, cl_cost, cl_time, bf_sol, bf_cost, bf_time 
        _Q = convert_qubo_to_Q(qubo, nn**2)
    
    print("\t\t"+colors.BOLD+colors.OKGREEN+"   PROBLEM BUILDED"+colors.ENDC+"\n\n\t\t"+colors.BOLD+colors.OKGREEN+"   START ALGORITHM"+colors.ENDC+"\n")
    
    if NPP:
        print("["+colors.BOLD+colors.OKCYAN+"S"+colors.ENDC+f"] {S}")

    start = time.time()
    z, r_time = solver.solve(d_min = 70, eta = 0.01, i_max = 150, k = 5, lambda_zero = 3/2, n = nn if NPP or QAP else nn ** 2 , N = 10, N_max = 100, p_delta = 0.1, q = 0.2, topology = 'pegasus', Q = _Q, log_DIR = log_DIR, sim = False)
    conv = datetime.timedelta(seconds=int(time.time() - start))

    min_z = solver.function_f(_Q,z).item()
    print("\t\t\t"+colors.BOLD+colors.OKGREEN+"RESULTS"+colors.ENDC+"\n")
    string = str()
    if nn < 16:
        string += log_write("Z",z)
    else:
        string += log_write("Z","Too big to print, see "+_DIR+"_solution.csv for the complete result")
    string += log_write("fQ",round(min_z,2))


    if NPP:
        diff2 = (c**2 + 4*min_z)
        string += log_write("c",c) + log_write("C",c**2) + log_write("DIFF", round(diff2,2)) + log_write("diff",np.sqrt(diff2))
        csv_write(DIR=_DIR+"_solution.csv", l=["c","c**2","diff**2","diff","S", "z", "Q"])
        csv_write(DIR=_DIR+"_solution.csv", l=[c,c**2,diff2,np.sqrt(diff2),S,z, _Q  if nn < 5 else "too big"])
        
    elif QAP:
        string += log_write("y",y) + log_write("Penalty",penalty) + log_write("Difference",round(y+min_z, 2)) #difference = y+minimum
        csv_write(DIR=_DIR+"_solution.csv", l=["problem","y","penalty","difference (y+minimum)", "z", "Q" ])
        csv_write(DIR=_DIR+"_solution.csv", l=[name,y,penalty,y+min_z,np.atleast_2d(z).T,_Q])

    else:
        DW = dict()
        DW['type'] = 'QALS'
        DW['response'] = z
        res = np.split(z,nn)
        valid = True
        fix_sol = list()
        for split in res:
            if np.count_nonzero(split == 1) != 1:
                valid = False
            where = str(np.where(split == 1))
            if str(np.where(split == 1)) in fix_sol:
                valid = False
            else:
                fix_sol.append(where)
        if not valid:
            string += "["+colors.BOLD+colors.FAIL+"ERROR"+colors.ENDC+"] Result is not valid.\n"
            DW['fixsol'] = list(tsp.fix_solution(z, True))
            string += "["+colors.BOLD+colors.WARNING+"VALID"+colors.ENDC+"] Validation occurred \n"
        else:
            DW['fixsol'] = []

        DW['fixcost'] = round(tsp.calculate_cost(tsp_matrix, DW['fixsol']), 2)
        DW['sol'] = tsp.binary_state_to_points_order(z)
        DW['cost'] = tsp.calculate_cost(tsp_matrix, DW['sol'])
        DW['rtime'] = r_time
        DW['ttime'] = conv

        tsp.write_TSP_csv(df, DW)

        #pd.DataFrame(["QALS", sol, cost, list(fix_sol), fix_cost, r_time, conv, list(z)]).to_csv(_DIR+"_solution.csv", mode='a',index=False)
        df.to_csv(_DIR+"_solution.csv")
        #csv_write(DIR=_DIR+"_solution.csv", l=["QALS", sol, cost, list(fix_sol), fix_cost, r_time, conv, list(z)])
    
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