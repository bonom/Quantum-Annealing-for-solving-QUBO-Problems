from QA4QUBO import matrix, vector, solver
import numpy as np

n = 10
max_range = 20

S = vector.generate_S(n, max_range)
_Q, c = matrix.generate_QUBO_problem(S)
#for row in _Q:
#    print(row)

"""
def matrix_to_dict(matrix, nodelist):
    n = len(matrix)
    m_t_ret = dict()
    j_max = 0
    for i in range(n):
        keys = list(nodelist.keys())
        key = keys[i]
        if matrix[i][i] != 0:
            m_t_ret[key,key] = int(matrix[i][i])
    
    for i in range(n):
        j = 0
        keys = list(nodelist.keys())
        key = keys[0]
        while j < j_max:
            #print(f"[{i}.{j}], [{j_max}] --> {matrix[i][j]}, \n!!!{nodelist}!!!")
            if matrix[i][j] != 0:
                values = list(nodelist[key])
                while(len(values) == 0):
                    del nodelist[key]
                    keys = list(nodelist.keys())
                    if len(keys) != 0:
                        key = keys[0]
                        values = list(nodelist[key])
                    else:
                        print(f"We have a problem from {i}.{j}")
                        exit()
                        
                m_t_ret[key,values[0]] = int(matrix[i][j])
                nodelist[key].remove(values[0])
            j += 1
        j_max += 1

    return m_t_ret
"""
def matrix_to_dict(matrix, nodelist):
    n = len(matrix)
    m_t_ret = dict()
    for i in range(n):
        keys = list(nodelist.keys())
        key = keys[i]
        if matrix[i][i] != 0:
            m_t_ret[key,key] = matrix[i][i]

    return m_t_ret

def get_active(sampler, n):
    nodes = dict()
    tmp = list(sampler.nodelist)
    nodelist = list()
    for i in range(n):
        nodelist.append(tmp[i])
        
    for i in nodelist:
        nodes[i] = list()
    
    for node_1,node_2 in sampler.edgelist:
        if node_1 in nodelist and node_2 in nodelist:
            nodes[node_1].append(node_2)
            nodes[node_2].append(node_1)

    if len(nodes) != n:
        print(f"Che facciamo? ho {len(nodes)} e {n}")
        exit()
    
    return nodes


from dwave.system.samplers import DWaveSampler
sampler = DWaveSampler(solver={'topology__type' : 'pegasus'})

nodes = get_active(sampler, n)
#print(nodes)
A = matrix.generate_pegasus(n)
Theta = solver.g(_Q, A, np.arange(n), 1)
print(Theta)
print(Theta[0])
print(Theta[0][0])
print(type(Theta))
tmp = matrix_to_dict(Theta, nodes.copy())

response = sampler.sample_qubo(tmp, num_reads=2)
print(response)
val = list(response.first.sample.values())
print(val)
print(f"\n{len(val) == n}")
