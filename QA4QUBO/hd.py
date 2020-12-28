import dimod
import hybrid
import numpy as np
import time

def run_annealer(theta, iteration, workflow):
    bqm = dimod.BinaryQuadraticModel({}, matrix_to_dict(theta), 0, dimod.BINARY)

    init_state = hybrid.State.from_problem(bqm)
    final_state = workflow.run(init_state).result()
    response = final_state.samples.first.sample.values()

    return np.atleast_2d(list(response)).T

def matrix_to_dict(theta):
    n = len(theta)
    d = dict()
    for i in range(n):
        for j in range(n):
            d[i, j] = theta[i][j]

    return d
    