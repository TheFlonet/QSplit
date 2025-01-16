import os
import dimod
from util import normalize_qubo
import numpy as np
import json

def main():
    paths = [f for f in os.listdir('../dataset') if f.endswith('.lp')]
    print('Path loaded')
    problems = [(dimod.lp.load('../dataset/' + p), p) for p in paths]
    print('Problem loaded')
    qubos = [(dimod.cqm_to_bqm(p[0])[0].to_qubo(), p[1]) for p in problems]
    print('Problem converted to QUBO')
    qubos = [(i[0][0], i[0][1], i[1], i[2]) for i in [(normalize_qubo(q[0][0]), q[0][1], q[1]) for q in qubos]]
    print('Problem normalized')
    for qubo_dict, num_vars, offset, problem_name in qubos:
        mat = [[0 for _ in range(num_vars)] for _ in range(num_vars)]
        for (a, b), v in qubo_dict.items():
            mat[a][b] = v
        with open(f'../dataset/qubo_instances/{problem_name[:-3]}.json', 'w') as f:
            json.dump({
                "offset": offset,
                "qubo_matrix": mat
            }, f)
    print('Done')

if __name__=='__main__':
    main()