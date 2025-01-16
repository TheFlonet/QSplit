import logging
import time
from dwave.system import LeapHybridSampler, EmbeddingComposite, DWaveSampler
from dwave.samplers import SimulatedAnnealingSampler
from dwave_qbsolv import QBSolv
from dotenv import load_dotenv
from qsplit.QSplitSampler import QSplit
from qsplit.QUBO import QUBO
import pandas as pd
import json

def runner(kind: str, qubo: QUBO, q_cut: int = 64):
    s = time.time()
    if kind == 'hybrid':
        sol = LeapHybridSampler().sample_qubo(qubo.qubo_dict, offset=qubo.offset).to_pandas_dataframe()
    elif kind == 'qbsolv_sa':
        sol = QBSolv().sample_qubo(qubo.qubo_dict, offset=qubo.offset).to_pandas_dataframe()
    elif kind == 'qbsolv_qpu':
        sol = QBSolv().sample_qubo(qubo.qubo_dict, solver=EmbeddingComposite(DWaveSampler()), 
                                   offset=qubo.offset, solver_limit=64).to_pandas_dataframe()
    elif kind == 'sa':
        sol = SimulatedAnnealingSampler().sample_qubo(qubo.qubo_dict, offset=qubo.offset, 
                                                      num_reads=10).to_pandas_dataframe()
    elif kind == 'qsplit_sa':
        sol = QSplit('sa', q_cut).sample_qubo(qubo)[0].solutions
    elif kind == 'qsplit_qpu':
        sol = QSplit('qpu', q_cut).sample_qubo(qubo)[0].solutions
    e = time.time()

    log.info(f'Running {kind} in {e-s:.2f}s, best energy: {sol["energy"].min():.2f}')

def sol_range(coeffs, offset):
    solver = SimulatedAnnealingSampler()
    min_val = solver.sample_qubo(coeffs, offset=offset, num_reads=100).to_pandas_dataframe()['energy'].min()
    max_val = -solver.sample_qubo({k: -v for k, v in coeffs.items()}, offset=offset, 
                                  num_reads=100).to_pandas_dataframe()['energy'].min()

    return round(min_val, 2), round(max_val, 2)

def main():
    load_dotenv()
    df = pd.read_csv('dataset/problems.csv')
    problems = list(df.itertuples(index=False, name=None))

    for idx, (problem_name, num_vars) in enumerate(problems):
        coeffs = {}
        with open(f'./dataset/qubo_instances/{problem_name}.json') as json_file:
            r = json.load(json_file)
            offset = r['offset']
            mat = r['qubo_matrix']
            for i, row in enumerate(mat):
                for j, cell in enumerate(row):
                    if cell != 0:
                        coeffs[(i, j)] = cell

        log.info(f'{idx+1}/{len(problems)}) {problem_name}')
        qubo_problem = QUBO(coeffs, offset=offset, cols_idx=list(range(num_vars)))
        log.info(f'Range {sol_range(coeffs, offset)}')
        runner('qsplit_sa', qubo_problem)
        runner('qbsolv_sa', qubo_problem)
        runner('sa', qubo_problem)


if __name__ == '__main__':
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]  %(message)s')
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fileHandler = logging.FileHandler('qsplit.log')
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)

    import warnings
    warnings.filterwarnings("ignore")

    main()
