from random import SystemRandom
random = SystemRandom()

import numpy as np

def generate_S(n, max):
    vect = [0 for i in range(n)]
    for index in range(n):
        vect[index] = random.randint(0,max)
    return vect
        
    