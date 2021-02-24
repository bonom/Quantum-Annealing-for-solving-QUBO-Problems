from os import urandom as _urandom
import random
BPF = 53        # Number of bits in a float
RECIP_BPF = 2**-BPF


for i in range(10):
    print(random.SystemRandom().randint(0, 1000))

print("after")
for i in range(10):
    print((int.from_bytes(_urandom(7), 'big') >> 3) * RECIP_BPF)