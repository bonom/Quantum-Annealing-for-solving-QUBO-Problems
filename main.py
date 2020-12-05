from QA4QUBO import matrix, solver, vector
import sys

def main(_n):
    if n == 8:
        S = [25, 7, 13, 31, 42,17, 21,10]
    elif n == 16:
        S = [1,2,3,4,5,6,7,8,9,8,7,6,6,5,4,5]
    else:
        S = vector.generate_S(_n)
    _Q = matrix.generate_QUBO_problem(S)
    _A = matrix.generate_chimera(1, 1)
    print(f" --- Problem start ---\n{_Q}\n\n{_A}\n ---------------------")
    print(solver.solve(d_min = 30, eta = 0.01, i_max = 3000, k = 1, lambda_zero = 1.0, n = _n, N = 8, N_max = 50, p_delta = 0.2, q = 0.1, A = _A, Q = _Q))



if __name__ == '__main__':
    try:
        n = int(sys.argv[1])
    except:
        n = 8
        pass
    main(n)