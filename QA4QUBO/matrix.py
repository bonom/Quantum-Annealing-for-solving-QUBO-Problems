import dwave_networkx as dnx

def generate_QUBO_problem(S):
    """
        Generate a QUBO problem (The number partitioning problem) from a vector S
    """
    n = len(S)
    c = 0
    for i in range(n):
        c += S[i]
    col_max = 0
    col = 0
    QUBO = [[0 for col in range(n)] for row in range(n)]
    for row in range(n):
        col_max += 1
        while col < col_max:
            if row == col:
                QUBO[row][col] = S[row]*(S[row]-c)
            else:
                QUBO[row][col] = S[row] * S[col]
                QUBO[col][row] = QUBO[row][col]
            col += 1
        col = 0
    return QUBO

def generate_chimera(r, c):
    """
        Sparse matrix chimera graph
    """
    G = dnx.chimera_graph(r, c)
    tmp = {}
    for n in G:
        tmp[n] = [nbr for nbr in G.neighbors(n) if nbr in G]
    n = len(tmp)
    rows = list()
    cols = list()
    for i in range(n):
        rows.append(i)
        cols.append(i)
        for element in tmp[i]:
            rows.append(i)
            cols.append(element)
    return list(zip(rows, cols))

