import dimod
import hybrid
import numpy as np
import datetime
import time

def annealer(theta, sampler, k):

    response = sampler.sample_qubo(theta, num_reads=k)
    
    return np.atleast_2d(list(response.first.sample.values())).T

def run_sim(theta, sampler, k):

    response = sampler.sample_qubo(theta, num_reads=k)
    
    return np.atleast_2d(list(response.first.sample.values())).T

def run_annealer(theta, sampler, k):
    # Run the annealer 4 times with theta matrix
    response = sampler.sample_qubo(theta, num_reads=k)
    
    # Samples are orderes from lowest energy to highest -> fist sample has lowest energy
    response = response.first.sample 

    return response