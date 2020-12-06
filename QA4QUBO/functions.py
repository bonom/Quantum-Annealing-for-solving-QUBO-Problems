#!/usr/bin/env python3

import random

def shuffle_vector(v):
    n = len(v)
    
    for i in range(n-1, 0, -1):
        j = random.randint(0,i) 
        v[i], v[j] = v[j], v[i]

def make_decision(probability):
    return random.random() < probability 