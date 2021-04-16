import dwave.inspector as inspector
from dwave.system import DWaveSampler, EmbeddingComposite

h = {'a': -1, 'b': 2, 'c': 1}
J = {('a', 'b'): 1.5, ('a', 'c'): -1, ('b', 'c'): 3}

sampler = DWaveSampler()
sampleset = sampler.sample_ising(h, J, num_reads=5)

inspector.show(sampleset)