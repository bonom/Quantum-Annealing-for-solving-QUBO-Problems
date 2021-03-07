import numpy as np
import time
import datetime

def annealer(theta, sampler, k, time=False):
    if time:
        start = time.time()

    response = sampler.sample_qubo(theta, num_reads=k) 
    
    if time:
        print(f"Time: {time.time()-start}")
    
    return np.atleast_2d(list(response.first.sample.values())).T