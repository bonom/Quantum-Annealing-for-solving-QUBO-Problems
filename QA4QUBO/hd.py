import dimod
import hybrid
from dwave.system.samplers import LeapHybridSampler
import numpy as np
import time


def run_annealer_hybrid(theta):
    sampler = LeapHybridSampler()
    response = sampler.sample_qubo(theta)
    response = response.first.sample.values()

    return np.atleast_2d(list(response)).T

def to_matrix(nums):
    theta = [[0 for col in range(len(nums))] for row in range(len(nums))]
    c = sum(nums)

    for i in range(len(nums)):
        for j in range(len(nums)):
            if i != j: theta[i][j] = nums[i] * nums[j]
            else: theta[i][i] = nums[i] * (nums[i] - c)

    return np.array(theta)

def matrix_to_dict(theta):
    n = len(theta)
    d = dict()
    for i in range(n):
        for j in range(n):
            d[i, j] = theta[i][j]

    return d

def fQ(theta, sol):
    return ((np.atleast_2d(sol).T).dot(theta)).dot(sol)
            
nums = [3, 7, 4, 10, 4, 2, 10, 1, 7, 3]
theta = to_matrix(nums)

start = time.time()
sol = run_annealer_hybrid(matrix_to_dict(theta))
end = time.time()

print(str(end - start) + "s")
print(fQ(theta, sol))