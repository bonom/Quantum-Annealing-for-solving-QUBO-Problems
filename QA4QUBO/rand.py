from os import urandom as _urandom
import random
BPF = 53        # Number of bits in a float
RECIP_BPF = 2**-BPF

def randint(lb:int,ub:int):
    return random.SystemRandom().randint(lb, ub)

def randfloat():
    return ((int.from_bytes(_urandom(64), 'big') >> 3) * 2**-53)

print((int.from_bytes(_urandom(7), 'big') >> 3) * RECIP_BPF)
print(randint(1,10))
print(randfloat)