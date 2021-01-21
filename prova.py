from QA4QUBO import matrix, vector
import dwave_networkx as dnx
import networkx as nx

def matrix_to_dict(matrix, active):
    n = len(matrix)
    j_max = 0
    j = 0
    m_t_dict = dict()
    for i in range(0,n):
        for j in range(0,n):
            if matrix[i][j] != 0:
                m_t_dict[active[i],active[j]] = matrix[i][j]
                #m_t_dict[active[j],active[i]] = matrix[j][i]
            
    return m_t_dict

#G = dnx.pegasus_graph(16, fabric_only=True)
#print(nx.to_dict_of_lists(G))

n = 10
max_range = 10

S = vector.generate_S(n, max_range)
_Q, c = matrix.generate_QUBO_problem(S)

###SOLUZIONE 1
"""
from dwave_qbsolv import QBSolv

response = QBSolv().sample_qubo(Q, num_reads=2)

print("samples="+str(list(response.samples())))
print("energies="+str(list(response.data_vectors['energy'])))
print("\n")
print(response.first.sample.values())
"""


###SOLUZIONE 2

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from minorminer import find_embedding

solver = DWaveSampler(solver={'topology__type' : 'pegasus'})

dim = 40
sparse = list()
matrix = [[0 for i in range(0, dim)] for i in range(0, dim)]
for coupler in solver.edgelist:
    u, v = coupler
    if u < dim and v < dim:
        matrix[u][v] = 1
        sparse.append(coupler)
"""
for i in range(dim):
    print(matrix[i])

print(sparse)
"""    

qubit_attivi = list()
for qubit in solver.nodelist:
    qubit_attivi.append(qubit)

print(len(qubit_attivi))

for qubit in range(0,5729):
    if qubit not in solver.nodelist:
        print(qubit, end =', ')
exit()
Q = matrix_to_dict(_Q, qubit_attivi)

print(Q)

sampler = DWaveSampler()

response = sampler.sample_qubo(Q, num_reads=2)

print(list(response.first.sample.values()))
