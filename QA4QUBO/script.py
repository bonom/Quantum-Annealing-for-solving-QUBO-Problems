import dimod
import hybrid
import numpy as np

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

def annealer(theta, sampler):
    
    if isinstance(theta, dict):
        pass
    elif isinstance(theta, np.ndarray) or isinstance(theta, list):
        theta = matrix_to_dict(theta)
    else:
        print(f"[!] Error -- I can't understand what type is {type(theta)}, only <class 'dict'>, <class 'numpy.ndarray'> or <class 'list'> admitted")
        raise TypeError
    
    response = sampler.sample_qubo(theta, num_reads=4)

    return np.atleast_2d(list(response.first.sample.values())).T

