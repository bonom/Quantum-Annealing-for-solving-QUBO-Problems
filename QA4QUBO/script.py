import dimod
import hybrid
import numpy as np
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

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
    vector = list()
    for i in range(n):
        vector.append(dic[i])

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

def annealer(theta):
    
    if isinstance(theta, dict):
        pass
    elif isinstance(theta, np.ndarray) or isinstance(theta, list):
        theta = matrix_to_dict(theta)
    else:
        print(f"[!] Error -- I can't understand what type is {type(theta)}, only <class 'dict'> or <class 'numpy.ndarray'> admitted")
        raise TypeError
    
    sampler = DWaveSampler()
    sampler = EmbeddingComposite(sampler)
    response = sampler.sample_qubo(theta, num_reads=4)

    return np.atleast_2d(list(response.first.sample.values())).T

def main(n):
    ls = list()
    for i in range(n):
        ls.append(np.random.randint(low=-100, high=100)/10)
    Q = generate_QUBO_problem(ls)
    if n <= 16:
        print(f"{Q}")
    else:
        print(" -- New Q created! --")
    # Define the workflow 
    """
    iteration = hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(),
        hybrid.EnergyImpactDecomposer(size=2)
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()
    ) | hybrid.ArgMin()
    
    workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=1)
    """
    import time
    time_s = time.time()
    print(annealer(Q))
    print(f"{time.time() - time_s} secondi")

if __name__ == '__main__':
    from QA4QUBO.matrix import generate_QUBO_problem
    main(8)
