import dimod
import hybrid
import numpy as np
import datetime
import time

def annealer(theta, sampler, k):

    response = sampler.sample_qubo(theta, num_reads=k)
    
    return np.atleast_2d(list(response.first.sample.values())).T