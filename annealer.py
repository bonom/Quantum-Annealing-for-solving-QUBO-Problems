import dimod
import hybrid
import time
import numpy as np

def function_f(Q, x):
    return ((np.atleast_2d(x).T).dot(Q)).dot(x)

def dict_to_vector(dic):
    """
        Convert a <class 'dict'> to a <class 'numpy.ndarray'>

        Args:
            dic:
                A dictionary to convert (must be <class 'dict'>)
        
        Returns:
            A <class 'numpy.ndarray'> vector

    """
    n = len(dic)
    vector = np.zeros((n))
    for i in range(n):
        vector[i] = dic[i]

    return np.atleast_2d(vector).T

def dict_to_matrix(dic):
    """
        Convert a <class 'dict'> to a <class 'numpy.ndarray'>

        Args:
            dic:
                A dict to convert (must be <class 'dict'>)
        
        Returns:
            A <class 'numpy.ndarray'> version of the <class 'dict'>

    """
    n = int(np.sqrt(len(dic)))
    matrix = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = dic[i,j]
    
    return matrix

def matrix_to_dict(matrix):
    """
        Convert a <class 'numpy.ndarray'> to a <class 'dict'>

        Args:
            matrix:
                A matrix to convert (must be <class 'numpy.ndarray'>)
        
        Returns:
            A dict version of the <class 'numpy.ndarray'>

    """
    n = len(matrix)
    m_t_dict = dict()
    for i in range(n):
        for j in range(n):
            m_t_dict[i,j] = matrix.item(i,j)
    
    return m_t_dict

def solve(Q, ret_dict = False):
    """
        Solve QUBO problems with annealing

        Args:
            Q:
                A matrix Q to minimize (can be <class 'numpy.ndarray'> or <class 'dict'>)

            ret dict:
                Boolean variable (default is False) that decides if return a <class 'dict'> (True) or a <class 'numpy.ndarray'> (False)

        Returns:
                Minimization <class 'dict'> or <class 'numpy.ndarray'>

        Examples:
                >>> solve(matrix)
        
    """

    #start_time = time.time()
    # Construct the QUBO problem
    if isinstance(Q, dict):
        print("")
        bqm = dimod.BinaryQuadraticModel({}, Q, 0, dimod.SPIN)
        n = int(np.sqrt(len(Q)))
        #bqm.normalize([-1.0, 1.0])
    elif isinstance(Q, np.ndarray):
        new_Q = matrix_to_dict(Q)
        print("... E convertuta")
        bqm = dimod.BinaryQuadraticModel({}, new_Q, 0, dimod.SPIN)
        n = len(Q)
        #bqm.normalize([-1.0, 1.0])
    else:
        print(f"[!] Error -- I can't understand what type is {type(Q)}, only <class 'dict'> or <class 'numpy.ndarray'> admitted")
        raise TypeError
    #print(f"Creazione problema: {time.time()-start_time} secondi con n = {n}")
    #print(f"Questo è il bqm:\n{bqm.quadratic.values()}")
    #if (input("Enter to continue...")) in ['n']:
    #    exit()
    #start_time = time.time()
    # Define the workflow
    print("1")
    iteration = hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(),
        hybrid.EnergyImpactDecomposer(size=1)
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()
    ) | hybrid.ArgMin()
    #print(f"Creazione iteration: {time.time()-start_time} secondi, iteration = {iteration}")
    #start_time = time.time()
    workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=1)
    print("2")
    #print(f"Creazione workflow: {time.time()-start_time} secondi")
    #start_time = time.time()
    # Solve the problem
    #print(f"Sto valutando l'init_state...", end = '\r')
    init_state = hybrid.State.from_problem(bqm)
    #print(f"init_state = {init_state}")
    #print(f"Sto valutando il final state...", end = '\r')
    print("3")
    final_state = workflow.run(init_state).result()
    #print(f"\n\nfinal_state = {final_state}\n\n e: {final_state.samples}")
    #print(f"Sto convertendo il final state...", end = '\r')
    print("4")
    solution = final_state.samples.first.sample
    #print(f"Creazione soluzione: {time.time()-start_time} secondi")
    print(f"Solution of Q \n{dict_to_vector(solution)}")
    # Print results
    #print("Solution: sample={.samples.first}".format(final_state))
    print(f"Vettore soluzione che porta f = {function_f(Q,dict_to_vector(solution))}")

    if(ret_dict):
        return solution
    return dict_to_vector(solution)


def main(n):
    """
    j_max = 0
    j = 0
    Q = dict()
    for i in range(n):
        j_max += 1
        while j < j_max:
            print(f"Creazione matrice... {int(i/n)}% -- {int(j/n)}%", end = '\r')
            if i == j:
                Q[i,j] = 0
            else:
                Q[i,j] = (np.random.randint(low=-100, high=100)/10)
            Q[j,i] = Q[i,j]
            j += 1
        j = 0
    """
    j_max = 0
    j = 0
    Q = np.zeros((n, n))
    for i in range(n):
        j_max += 1
        while j < j_max:
            Q[i][j] = (np.random.randint(low=-100, high=100)/100.0)
            Q[j][i] = Q[i][j]
            j += 1
        j = 0
    print("Matrice creata")
    #print(f"Creata Q di dimensione ({np.sqrt(len(Q))}, {np.sqrt(len(Q))}) ed è:\n{dict_to_matrix(Q)}")
    """
    Q = np.matrix([
         [0,  0.2,  9.7,  9.5, -4.6, -8.2,  4.7, -3.4, -3.5,    6,  0.2,  6.3,  8.6,  8.7,  7.8,  3.9],
       [0.2,    1, -0.1, -7.1, -5.5,  9.3,  6.8, -7.7,  6.7,   -3, -3.5, -6.9,    9,  4.6, -6.2, -0.4],
       [9.7, -0.1,    2, -1.9, -0.3,  9.8, -2.3, -0.9, -9.9, -5.5, -6.2, -3.2,    1,  4.8,    2, -7.7],
       [9.5, -7.1, -1.9,    3,  3.3,  0.7,  5.1, -7.9, -9.5,    8, -8.3,  9.8,  4.7,  9.1,  6.4,  6.7],
      [-4.6, -5.5, -0.3,  3.3,    4, -4.5,  4.6,  5.7,  -10,  3.4, -4.8, -1.9,   -7,   10,  0.8,  2.2],
      [-8.2,  9.3,  9.8,  0.7, -4.5,    5, -9.9,  5.4,    6, -8.1, -8.7,  0.7,  3.9, -6.4,    9, -5.5],
       [4.7,  6.8, -2.3,  5.1,  4.6, -9.9,    6,  3.7, -8.9,  -10,  1.6,  7.9,  4.8, -8.8,  6.9,  1.2],
      [-3.4, -7.7, -0.9, -7.9,  5.7,  5.4,  3.7,    7,    8, -7.7, -9.3, -1.4,  7.4,  4.1,  3.8, -9.5],
      [-3.5,  6.7, -9.9, -9.5,  -10,    6, -8.9,    8,    8,  2.1,  3.7,  1.3, -5.8, -1.2, -8.4,  5.2],
         [6,   -3, -5.5,    8,  3.4, -8.1,  -10, -7.7,  2.1,    9,  0.7,  8.1, -4.2,  9.7,  6.7,  9.9],
       [0.2, -3.5, -6.2, -8.3, -4.8, -8.7,  1.6, -9.3,  3.7,  0.7,   10,  9.2,  0.4,    6,  9.3,    7],
       [6.3, -6.9, -3.2,  9.8, -1.9,  0.7,  7.9, -1.4,  1.3,  8.1,  9.2,   11,  3.8,    4,  8.3,  0.6],
       [8.6,    9,    1,  4.7,   -7,  3.9,  4.8,  7.4, -5.8, -4.2,  0.4,  3.8,   12, -9.9,  1.2, -2.1],
       [8.7,  4.6,  4.8,  9.1,   10, -6.4, -8.8,  4.1, -1.2,  9.7,    6,    4, -9.9,   13,  5.9,  9.8],
       [7.8, -6.2,    2,  6.4,  0.8,    9,  6.9,  3.8, -8.4,  6.7,  9.3,  8.3,  1.2,  5.9,   14, -9.8],
       [3.9, -0.4, -7.7,  6.7,  2.2, -5.5,  1.2, -9.5,  5.2,  9.9,    7,  0.6, -2.1,  9.8, -9.8,   15]])
    """
    start_time = time.time()
    solve(Q)
    print(f"Terminato in {time.time()- start_time}")

if __name__ == '__main__':
    
    main(2048)
