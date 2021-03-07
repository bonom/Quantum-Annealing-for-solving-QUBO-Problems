import numpy as np

def run_annealer_hybrid(theta, sampler):

    response = sampler.sample_qubo(theta)

    return np.atleast_2d(list(response.first.sample.values())).T
    