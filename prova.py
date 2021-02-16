import sys

def read_integers(filename:str):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]

file_it = iter(read_integers("QA4QUBO/npp/Esc16.txt"))

n = next(file_it)

A = [[next(file_it) for j in range(n)] for i in range(n)]
B = [[next(file_it) for j in range(n)] for i in range(n)]

print(A)
print(B)