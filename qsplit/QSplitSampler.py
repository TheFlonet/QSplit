import logging
from collections import defaultdict
from typing import Tuple, List, Any
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import EmbeddingComposite, DWaveSampler
import numpy as np
import pandas as pd
from qsplit.QUBO import QUBO
import asyncio

log = logging.getLogger('qsplit')


class QSplit:
    def __init__(self, sampler: str, cut_dim: int):
        self.sampler = sampler
        self.cut_dim = cut_dim

    async def async_sample_qubo(self, qubo: QUBO) -> Tuple[QUBO, float]:
        if qubo.problem_size <= self.cut_dim or np.count_nonzero(qubo.qubo_matrix) <= self.cut_dim*(self.cut_dim+1)/2:
            if len(qubo.qubo_dict) == 0:
                all_indices = sorted(list(set(qubo.rows_idx).union(qubo.cols_idx)))
                data = [[np.nan for _ in range(len(all_indices) + 1)]]
                qubo.solutions = pd.DataFrame(data, columns=all_indices + ['energy'])
                q_time = 0
            else:
                if self.sampler == 'sa':
                    sampleset = SimulatedAnnealingSampler().sample_qubo(qubo.qubo_dict, num_reads=10, offset=qubo.offset)
                    q_time = 0
                elif self.sampler == 'qpu':
                    sampleset = EmbeddingComposite(DWaveSampler()).sample_qubo(qubo.qubo_dict, num_reads=10, offset=qubo.offset)
                    q_time = sampleset.info['timing']['qpu_access_time'] / 1e6
                else:
                    raise Exception('Unsupported solver')
                res = (sampleset.to_pandas_dataframe().drop(columns=['num_occurrences']).drop_duplicates()
                       .sort_values(by='energy', ascending=True))
                qubo.solutions = res[res['energy'] == min(res['energy'])]
            return qubo, q_time

        sub_problems = self.__split_problem(qubo)
        solutions, total_q_time = [], 0

        results = await asyncio.gather(*[self.async_sample_qubo(s) for s in sub_problems])

        solutions = []
        for solution, qpu_time in results:
            solutions.append(solution)
            total_q_time += qpu_time

        return self.__aggregate_solutions(solutions, total_q_time, qubo)

    def sample_qubo(self, qubo: QUBO) -> Tuple[QUBO, float]:
        return asyncio.run(self.async_sample_qubo(qubo))

    def __aggregate_solutions(self, solutions: List[QUBO], prev_q_time: float, qubo: QUBO) -> Tuple[QUBO, float]:
        # Aggregate upper-left qubo with lower-right
        starting_sols = self.__combine_ul_lr(solutions[0], solutions[2])
        # Set missing columns in upper-right qubo to NaN
        ur_qubo_filled = self.__fill_with_nan(starting_sols.columns, solutions[1].solutions)
        # Search the closest assignments between upper-right qubo and merged solution (UL and LR qubos)
        closest_df = self.__get_closest_assignments(starting_sols, ur_qubo_filled)
        # Combine
        combined_df = pd.DataFrame([self.__combine_rows(row1, row2) for (_, row1), (_, row2) in
                                    zip(starting_sols.iterrows(), closest_df.iterrows())],
                                   columns=starting_sols.columns)
        # Conflicts resolution
        qubo.solutions, local_q_time = self.__local_search(combined_df, qubo)
        qubo.solutions = qubo.solutions.reset_index(drop=True).drop_duplicates().nsmallest(n=10, columns='energy')

        return qubo, prev_q_time + local_q_time

    @staticmethod
    def __combine_rows(row1: pd.Series, row2: pd.Series) -> List[float | Any]:
        combined_row = []
        for col in row1.index:
            val1, val2 = row1[col], row2[col]
            if col == 'energy':
                if (np.nan in combined_row) or (np.isnan(val1) and np.isnan(val2)):
                    combined_row.append(np.nan)
                elif np.isnan(val1):
                    combined_row.append(val2)
                elif np.isnan(val2):
                    combined_row.append(val1)
                else:
                    combined_row.append(val1 + val2)
            else:
                if pd.isna(val2) and not pd.isna(val1):
                    combined_row.append(val1)
                elif pd.isna(val1) and not pd.isna(val2):
                    combined_row.append(val2)
                elif val1 == val2:
                    combined_row.append(val1)
                else:
                    combined_row.append(np.nan)
        return combined_row

    def __get_closest_assignments(self, starting_sols: pd.DataFrame, ur_qubo_filled: pd.DataFrame) -> pd.DataFrame:
        closest_rows = []
        for _, row in starting_sols.iterrows():
            distances = []
            for _, sol_row in ur_qubo_filled.iterrows():
                distance = self.__nan_hamming_distance(row.values, sol_row.values)
                distances.append(distance)
            closest_idx = np.argmin(distances)
            to_append = ur_qubo_filled.iloc[closest_idx].copy()
            if np.any(to_append.isna()):
                to_append['energy'] = np.nan
            closest_rows.append(to_append)
        return pd.DataFrame(closest_rows).reset_index(drop=True)

    @staticmethod
    def __nan_hamming_distance(a: np.ndarray, b: np.ndarray) -> float | Any:
        mask = ~np.isnan(a) & ~np.isnan(b)
        if np.sum(mask) == 0:
            return np.inf
        return np.sum(a[mask] != b[mask]) / np.sum(mask)

    def __local_search(self, df: pd.DataFrame, qubo: QUBO) -> Tuple[pd.DataFrame, float]:
        q_time = 0
        for i in range(len(df)):
            no_energy = df.loc[i].drop('energy')

            if not np.any(np.isnan(no_energy.values)):
                df.loc[i, 'energy'] = no_energy.values.T @ qubo.qubo_matrix @ no_energy.values
            else:
                nans = no_energy[np.isnan(no_energy)]
                qubo_nans = defaultdict(int)
                all_zero = True
                for row_idx in nans.index:
                    for col_idx in nans.index:
                        if (row_idx, col_idx) in qubo.qubo_dict:
                            qubo_nans[(row_idx, col_idx)] = qubo.qubo_dict[(row_idx, col_idx)]
                            all_zero = False
                        else:
                            qubo_nans[(row_idx, col_idx)] = 0
                if all_zero:
                    df.loc[i, nans.index] = 0
                    no_energy = df.loc[i].drop('energy')
                    cut = qubo.qubo_matrix.shape[0]
                    df.loc[i, 'energy'] = no_energy.values[:cut].T @ qubo.qubo_matrix @ no_energy.values[:cut]
                else:
                    if self.sampler == 'sa':
                        nans_sol = SimulatedAnnealingSampler().sample_qubo(qubo_nans, num_reads=5, offset=qubo.offset)
                    elif self.sampler == 'qpu':
                        nans_sol = EmbeddingComposite(DWaveSampler()).sample_qubo(qubo_nans, num_reads=5, offset=qubo.offset)
                        q_time += nans_sol.info['timing']['qpu_access_time'] / 1e6
                    else:
                        raise Exception('Unsupported solver')
                    nans_sol = nans_sol.to_pandas_dataframe().sort_values(by='energy', ascending=True).iloc[0]
                    if np.isnan(df.loc[i, 'energy']):
                        df.loc[i, 'energy'] = 0
                    df.loc[i, nans.index] = nans_sol.drop('energy')
                    df.loc[i, 'energy'] += nans_sol['energy']

        return df, q_time

    @staticmethod
    def __fill_with_nan(schema: pd.Index, df_to_fill: pd.DataFrame) -> pd.DataFrame:
        missing_columns = set(schema) - set(df_to_fill.columns)
        for col in missing_columns:
            df_to_fill[col] = np.nan
        return df_to_fill[schema]

    def __combine_ul_lr(self, ul: QUBO, lr: QUBO) -> pd.DataFrame:
        all_indices = sorted(list(set(ul.rows_idx).union(lr.cols_idx)))
        ul.solutions['tmp'] = 1
        lr.solutions['tmp'] = 1
        merge = pd.merge(ul.solutions, lr.solutions, on='tmp')
        merge['energy'] = merge['energy_x'] + merge['energy_y']
        merge = merge.drop(['energy_x', 'energy_y', 'tmp'], axis=1)
        ul.solutions.drop('tmp', axis=1, inplace=True)
        lr.solutions.drop('tmp', axis=1, inplace=True)
        return self.__fill_with_nan(pd.Index(all_indices + ['energy']), merge)

    @staticmethod
    def __split_problem(qubo: QUBO) -> Tuple[QUBO, QUBO, QUBO]:
        """
            Returns 3 sub-problems in qubo form.
            The 3 sub-problems correspond to the matrices obtained by dividing the qubo matrix of the original problem
            in half both horizontally and vertically.
            The sub-problem for the sub-matrix in the bottom left corner is not given as this is always empty.
            The order of the results is:
            - Upper left sub-matrix,
            - Upper right sub-matrix,
            - Lower right sub-matrix.

            All sub-problems are converted to obtain an upper triangular matrix.
        """
        split_u_l = defaultdict(int)
        split_u_r = defaultdict(int)
        split_l_r = defaultdict(int)
        split_idx = qubo.problem_size // 2

        for k, v in qubo.qubo_dict.items():
            row, col = k[0], k[1]
            if row < split_idx and col < split_idx:
                split_u_l[k] = v
            elif row < split_idx <= col:
                split_u_r[k] = v
            elif row >= split_idx and col >= split_idx:
                split_l_r[k] = v
            else:
                raise ValueError(
                    'All values in the lower left matrix should be 0, so not present in the qubo dictionary')

        res = (QUBO(split_u_l, cols_idx=qubo.cols_idx[:split_idx], rows_idx=qubo.rows_idx[:split_idx], to_transform=False),
               QUBO(split_u_r, cols_idx=qubo.cols_idx[split_idx:], rows_idx=qubo.rows_idx[:split_idx]),
               QUBO(split_l_r, cols_idx=qubo.cols_idx[split_idx:], rows_idx=qubo.rows_idx[split_idx:], to_transform=False))

        return res
