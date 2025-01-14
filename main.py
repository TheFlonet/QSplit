import logging
import time
import dimod.lp
from dwave.system import LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
from dwave_qbsolv import QBSolv
from dotenv import load_dotenv
from qsplit.QSplitSampler import QSplit
from qsplit.QUBO import QUBO
import dimod
import pandas as pd
import os
from util import normalize_qubo

def runner(kind: str, qubo: QUBO, q_cut: int = 64):
    s = time.time()
    if kind == 'hybrid':
        sol = LeapHybridSampler().sample_qubo(qubo.qubo_dict, offset=qubo.offset).to_pandas_dataframe()
    elif kind == 'qbsolv_sa':
        sol = QBSolv().sample_qubo(qubo.qubo_dict, offset=qubo.offset).to_pandas_dataframe()
    elif kind == 'qbsolv_qpu':
        sol = QBSolv().sample_qubo(qubo.qubo_dict, solver='dw', offset=qubo.offset).to_pandas_dataframe()
    elif kind == 'sa':
        sol = SimulatedAnnealingSampler().sample_qubo(qubo.qubo_dict, offset=qubo.offset, num_reads=10).to_pandas_dataframe()
    elif kind == 'qsplit_sa':
        sol = QSplit('sa', q_cut).sample_qubo(qubo)[0].solutions
    elif kind == 'qsplit_qpu':
        sol = QSplit('qpu', q_cut).sample_qubo(qubo)[0].solutions
    e = time.time()

    log.info(f'Running {kind} in {e-s:.2f}s, best energy: {sol["energy"].min():.2f}')

def sol_range(coeffs, offset):
    solver = SimulatedAnnealingSampler()
    min_val = solver.sample_qubo(coeffs, offset=offset, num_reads=100).to_pandas_dataframe()['energy'].min()
    max_val = -solver.sample_qubo({k: -v for k, v in coeffs.items()}, offset=offset, num_reads=100).to_pandas_dataframe()['energy'].min()

    return round(min_val, 2), round(max_val, 2)

def main():
    load_dotenv()
    os.system('clear')
    df = pd.read_csv('dataset/problems.csv')
    problems = list(df.itertuples(index=False, name=None))

    for idx, p in enumerate(problems):
        coeffs, offset = dimod.cqm_to_bqm(dimod.lp.load('dataset/' + p[0]))[0].to_qubo()
        coeffs, qubo_size = normalize_qubo(coeffs)

        log.info(f'{idx+1}/{len(problems)}) {p[0]}')
        qubo_problem = QUBO(coeffs, offset=offset, cols_idx=list(range(qubo_size)))
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
