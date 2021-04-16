import numpy as np
import time
import datetime
import dwave.inspector as inspector
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod

def annealer(theta, sampler, k, time=False):
    if time:
        start = time.time()
    
    response = sampler.sample_qubo(theta, num_reads=k) 
    #inspector.show(response)
    #input("Proseguire?")
    if time:
        print(f"Time: {time.time()-start}")
    
    return np.atleast_2d(list(response.first.sample.values())).T