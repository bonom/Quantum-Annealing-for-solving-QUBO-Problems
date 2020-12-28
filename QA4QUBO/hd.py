from dwave.system.samplers import LeapHybridSampler
import numpy as np

def run_annealer(theta, sampler):
    
    response = sampler.sample_qubo(matrix_to_dict(theta))
    response = response.first.sample.values()

    return np.atleast_2d(list(response)).T

def matrix_to_dict(theta):
    n = len(theta)
    d = dict()
    for i in range(n):
        for j in range(n):
            d[i, j] = theta[i][j]

    return d
    