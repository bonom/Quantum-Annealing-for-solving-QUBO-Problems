from dwave_networkx.algorithms.tsp import traveling_salesperson_qubo
import numpy as np
import networkx as nx
import neal
import dimod

def main():
    G = nx.Graph()
    G.add_weighted_edges_from({(0, 1, .1), (0, 2, .9), (1, 2, .1)})
    d = traveling_salesperson_qubo(G) # doctest: +SKIP
    #[0, 1, 2, 3]
    indexes = dict()
    it = 0
    for i in range(3):
        for j in range(3):
            indexes[(i,j)] = it
            it += 1

    print(indexes)

    matrix = np.zeros((9,9))

    for key_1, key_2 in d:
        matrix[indexes[key_1],indexes[key_2]] = d[key_1,key_2]

    return matrix
    #

if __name__ == '__main__':
    print(main())