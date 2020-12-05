import random as rn

def generate_S(n):
    vect = [0 for i in range(n)]
    for index in range(n):
        vect[index] = (rn.randint(-1000,1000))/10
    return vect
        
    