import random
from .get_from_file import elements

tests = elements()

def shuffle_vector(v):
    n = len(v)
    for i in range(n-1, -1, -1):
        val = tests.pop_int()    
        j = int((val * i)/4000)  
        v[i], v[j] = v[j], v[i]
    

def make_decision(probability):
    return tests.pop_float() <= probability  