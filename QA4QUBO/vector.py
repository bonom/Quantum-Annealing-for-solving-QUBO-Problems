#!/usr/bin/env python3

import random as rn
import numpy as np

def generate_S(n, max):
    vect = [0 for i in range(n)]
    for index in range(n):
        vect[index] = rn.randint(0,max)
    return vect
        
    