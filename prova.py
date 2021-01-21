from QA4QUBO import matrix, vector
import dwave_networkx as dnx
import networkx as nx
from dwave.system.composites import EmbeddingComposite

#G = dnx.pegasus_graph(16, fabric_only=True)
#print(nx.to_dict_of_lists(G))

n = 3
max_range = 20

S = vector.generate_S(n, max_range)
_Q, c = matrix.generate_QUBO_problem(S)
for row in _Q:
    print(row)

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
"""
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from minorminer import find_embedding

def insert(i, j, val, nodelist, to_ret, nodi):
    keys = list(nodelist.keys())
    key = keys[i]
    value_list = list(nodelist[key])
    if len(value_list) == 0:
        del nodelist[key]
        if (i-1 < 0):
            insert(i, j, val, nodelist, to_ret, nodi)
        else:
            insert(i-1, j, val, nodelist, to_ret, nodi)
    else:
        print(f"Inserito {val} in [{key},{value_list[0]}]")
        to_ret[key, value_list[0]] = val
        if key not in nodi:
            nodi.append(key)
        if value_list[0] not in nodi:
            nodi.append(value_list[0])
        nodelist[key].remove(value_list[0])
        

def matrix_to_dict(matrix, nodelist):
    n = len(matrix)
    j_max = 0
    m_t_dict = dict()
    nodi = list()
    for i in range(0,n):
        j_max += 1
        j = 0
        while j < j_max:
            print(f"{i}. - {j} --> ")
            if matrix[i][j] != 0:
                insert(i,j,matrix[i][j],nodelist,m_t_dict,nodi)
            j += 1
            
    nodi.sort()
    print(nodi)      
    return m_t_dict

solver = DWaveSampler(solver={'topology__type' : 'pegasus'})

nodes = [[]for i in range(int(max(solver.nodelist))+1)]
nodelist = dict()
for i in solver.nodelist:
    nodelist[i] = list()
for node_1,node_2 in solver.edgelist:
    nodelist[node_1].append(node_2)
    nodelist[node_2].append(node_1)
    
Q = matrix_to_dict(_Q, nodelist)
#exit()

response = solver.sample_qubo(Q, num_reads=2)
print(response)
val = list(response.first.sample.values())
print(len(val))
"""

#SOLUZIONE 3
from dwave.system.samplers import DWaveSampler

def matrix_to_dict(matrix, nodes):
    n = len(matrix)
    m_t_ret = dict()
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                m_t_ret[nodes[i], nodes[j]] = matrix[i][j]

    return m_t_ret


sampler = DWaveSampler(solver={'topology__type' : 'pegasus'})

nodes = list()
for node in sampler.nodelist:
    nodes.append(node)

Q = matrix_to_dict(_Q, nodes)

response = sampler.sample_qubo(Q, num_reads=2)
print(response)
val = list(response.first.sample.values())
print(len(val))
