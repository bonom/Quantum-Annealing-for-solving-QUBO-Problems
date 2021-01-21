from QA4QUBO import matrix, vector
import dwave_networkx as dnx
import networkx as nx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from minorminer import find_embedding

solver = DWaveSampler(solver={'topology__type' : 'pegasus'})

n = 10
max_range = 10

S = vector.generate_S(n, max_range)
_Q, c = matrix.generate_QUBO_problem(S)

def create_A(solver=DWaveSampler(solver={'topology__type' : 'pegasus'})):
    for 