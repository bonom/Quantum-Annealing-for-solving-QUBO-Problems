import dimod
import hybrid
import time
import numpy as np

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
    j_max = 0
    j = 0
    m_t_dict = dict()
    for i in range(n):
        j_max += 1
        while j < j_max:
            m_t_dict[i,j] = matrix[i][j]
            j += 1
        j = 0
    
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
    
    # Construct the QUBO problem
    if isinstance(Q, dict):
        bqm = dimod.BinaryQuadraticModel({}, Q, 0, dimod.SPIN)
    elif isinstance(Q, np.ndarray):
        new_Q = matrix_to_dict(Q)
        bqm = dimod.BinaryQuadraticModel({}, new_Q, 0, dimod.SPIN)
    else:
        print(f"[!] Error -- I can't understand what type is {type(Q)}, only <class 'dict'> or <class 'numpy.ndarray'> admitted")
        raise TypeError
    
    # Define the workflow --- DO NOT TOUCH
    iteration = hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(),
        hybrid.EnergyImpactDecomposer(size=1)
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()
    ) | hybrid.ArgMin()
    
    workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=1)

    # Solve the problem
    init_state = hybrid.State.from_problem(bqm)
    final_state = workflow.run(init_state).result()
    solution = final_state.samples.first.sample
    
    #print(f"Solution of Q = {solution}")
    # Print results
    #print("Solution: sample={.samples.first}".format(final_state))

    if(ret_dict):
        return solution
    return dict_to_vector(solution)


def main(n):
    j_max = 0
    j = 0
    Q = dict()
    for i in range(n):
        j_max += 1
        while j < j_max:
            Q[i,j] = np.random.randint(low=-10, high=10)
            Q[j,i] = Q[i,j]
            j += 1
        j = 0
    solve(Q)

if __name__ == '__main__':
    main(2048)
