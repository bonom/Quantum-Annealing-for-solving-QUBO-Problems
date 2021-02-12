#!/usr/bin/env python3

from QA4QUBO import matrix, vector, solver
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
qap = [f for f in listdir("QA4QUBO/tests/") if isfile(join("QA4QUBO/tests/", f))]
npp = [f for f in listdir("QA4QUBO/npp/") if isfile(join("QA4QUBO/npp/", f))]
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
    max_range = 1000
    dir = "output_"+str(_n)+"_"+ str(max_range)
    while(nok):
        try:
            with open("output/"+dir+".txt", "r") as file:
                pass
            max_range = int(MAX/(2**i))
            if(not max_range):
                exit("File output terminati")
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
    #S = [357, 0, 881, 284, 385, 662, 986, 666, 973, 887, 249, 919, 268, 623, 999, 386, 201, 9, 170, 767, 40, 450, 96, 216, 474, 649, 817, 509, 84, 159, 908, 761, 499, 119, 724, 951, 520, 973, 78, 238, 469, 995, 596, 382, 538, 937, 296, 984, 479, 452, 484, 119, 816, 301, 323, 684, 595, 406, 875, 872, 312, 970, 337, 459, 860, 387, 731, 885, 233, 488, 626, 539, 298, 595, 740, 392, 100, 670, 227, 523, 48, 534, 389, 608, 435, 969, 843, 387, 638, 597, 540, 173, 119, 800, 858, 849, 122, 738, 416, 941, 750, 224, 388, 699, 325, 893, 578, 214, 726, 258, 789, 31, 32, 473, 173, 719, 641, 623, 442, 198, 963, 787, 908, 184, 237, 190, 538, 15, 736, 874, 23, 717, 881, 632, 956, 508, 956, 729, 790, 261, 836, 82, 37, 354, 420, 970, 950, 326, 173, 571, 718, 923, 737, 286, 312, 596, 538, 790, 979, 878, 841, 395, 979, 25, 179, 778, 825, 169, 922, 200, 689, 952, 541, 383, 860, 643, 554, 524, 192, 112, 686, 619, 10, 571, 1000, 80, 74, 622, 742, 622, 387, 941, 715, 555, 32, 117, 94, 41, 298, 951, 577, 617, 917, 937, 298, 59, 553, 718, 41, 827, 805, 339, 275, 41, 363, 555, 421, 200, 802, 866, 769, 513, 49, 228, 280, 997, 747, 578, 298, 461, 469, 830, 877, 335, 370, 468, 953, 466, 280, 808, 736, 211, 637, 882, 59, 684, 357, 420, 353, 160, 598, 290, 66, 15, 446, 19, 483, 379, 747, 996, 238, 363, 455, 215, 6, 716, 751, 398, 493, 564, 563, 216, 479, 926, 257, 650, 227, 808, 182, 77, 388, 916, 963, 126, 182, 41, 919, 229, 507, 503, 860, 763, 683, 326, 310, 582, 698, 493, 727, 442, 977, 19, 280, 495, 20, 181, 952, 411, 999, 590, 170, 475, 185, 868, 110, 951, 276, 129, 876, 458, 917, 410, 11, 568, 906, 791, 970, 608, 613, 937, 303, 628, 460, 97, 923, 857, 991, 775, 673, 386, 993, 625, 247, 312, 877, 61, 444, 263, 420, 784, 725, 83, 383, 257, 118, 600, 7, 485, 558, 46, 680, 273, 880, 278, 736, 170, 330, 270, 461, 722, 166, 627, 325, 261, 946, 227, 306, 290, 395, 450, 91, 378, 102, 447, 705, 652, 728, 231, 510, 199, 836, 633, 478, 236, 118, 191, 820, 339, 615, 891, 952, 473, 793, 825, 123, 442, 83, 313, 162, 942, 289, 971, 205, 844, 134, 3, 767, 78, 166, 898, 22, 311, 806, 368, 41, 430, 389, 824, 720, 989, 664, 164, 370, 284, 735, 978, 368, 225, 662, 920, 204, 462, 478, 759, 564, 643, 139, 332, 334, 784, 622, 831, 47, 718, 743, 411, 779, 713, 891, 935, 814, 244, 739, 549, 793, 821, 312, 183, 560, 67, 328, 204, 674, 55, 590, 899, 244, 864, 371, 672, 234, 314, 727, 388, 715, 371, 605, 390, 482, 395, 36, 864, 923, 872, 145, 239, 832, 167, 474, 553]
    _Q, c = matrix.generate_QUBO_problem(S)

    string = " ---------- Problem start ----------\n"
    print(string)
    write(_DIR, string)
    
    if not QAP:
        string = "\n S = "+str(S)+"\n"
        print(string)
        write(_DIR, string)
    
    
    z = solver.solve(d_min = 70, eta = 0.01, i_max = 1000, k = 10, lambda_zero = 1, n = _n, N = 10, N_max = 100, p_delta = 0.1, q = 0.2, topology = 'pegasus', Q = _Q, DIR = _DIR, sim = True)
    
    min_z = solver.function_f(_Q,np.atleast_2d(z).T).item()
    
    string = "So far we found:\n- z - \n"+str(np.atleast_2d(z).T)+"\nand has minimum = "+str(min_z)+"\n"
    diff2 = (c**2 + 4*min_z)

    try:
        string += "c = "+str(c)+", c^2 = "+str(c**2)+", diff^2 = "+str(diff2)+", diff = "+str(np.sqrt(diff2))+"\n"
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