import dimod
import hybrid
import numpy as np

def solve(Q, debug = False):
    # Construct a problem
    bqm = dimod.BinaryQuadraticModel({}, Q, 0, dimod.SPIN)

    # Define the workflow
    iteration = hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(),
        hybrid.EnergyImpactDecomposer(size=2)
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()
    ) | hybrid.ArgMin()
    workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=1)

    # Solve the problem
    init_state = hybrid.State.from_problem(bqm)
    final_state = workflow.run(init_state).result()

    solution = final_state.samples.first.sample

    print(f"Soluzione di Q = {solution}")
    # Print results
    #print("Solution: sample={.samples.first}".format(final_state))


def main(n = 4):
    j_max = 0
    j = 0
    Q = dict()
    for i in range(n):
        j_max += 1
        while j < j_max:
            Q[i,j] = np.random.randint(low=-10, high=10)
            Q[j,i] = Q[i,j]
            j += 1
        j = 0

if __name__ == '__main__':
    main()