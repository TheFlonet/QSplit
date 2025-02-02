import cProfile
import logging
from collections import defaultdict
import pstats
from typing import Tuple, List, Dict
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import EmbeddingComposite, DWaveSampler
import numpy as np
import pandas as pd
from qsplit.QUBO import QUBO
import asyncio

log = logging.getLogger('qsplit')
PROFILE = False


class QSplit:
    def __init__(self, sampler: str, cut_dim: int):
        self.sampler = sampler
        self.default_sampler = SimulatedAnnealingSampler()
        self.cut_dim = cut_dim

    def __get_result(self, qubo_dict: Dict, offset: float, reads: int, use_default: bool = False) -> Tuple[pd.DataFrame, float]:
        if use_default:
            sampleset = self.default_sampler.sample_qubo(qubo_dict, num_reads=reads, offset=offset)
            q_time = 0
        elif self.sampler == 'sa':
            sampleset = SimulatedAnnealingSampler().sample_qubo(qubo_dict, num_reads=reads, offset=offset)
            q_time = 0
        elif self.sampler == 'qpu':
            sampleset = EmbeddingComposite(DWaveSampler()).sample_qubo(qubo_dict, num_reads=reads, offset=offset)
            q_time = sampleset.info['timing']['qpu_access_time'] / 1e6
        else:
            raise Exception('Unsupported solver')
        
        sampleset = (sampleset.to_pandas_dataframe().drop(columns=['num_occurrences']).drop_duplicates()
                       .sort_values(by='energy', ascending=True))
        return sampleset, q_time

    async def async_sample_qubo(self, qubo: QUBO) -> Tuple[QUBO, float]:
        if qubo.problem_size <= self.cut_dim or np.count_nonzero(qubo.qubo_matrix) <= self.cut_dim*(self.cut_dim+1)/2:
            if len(qubo.qubo_dict) == 0:
                all_indices = sorted(list(set(qubo.rows_idx).union(qubo.cols_idx)))
                data = [[np.nan for _ in range(len(all_indices) + 1)]]
                qubo.solutions = pd.DataFrame(data, columns=all_indices + ['energy'])
                q_time = 0
            else:
                res, q_time = self.__get_result(qubo.qubo_dict, qubo.offset, 10)
                qubo.solutions = res[res['energy'] == res['energy'].min()]
            return qubo, q_time

        sub_problems = self.__split_problem(qubo)
        solutions, total_q_time = [], 0

        results = await asyncio.gather(*[self.async_sample_qubo(s) for s in sub_problems])

        solutions = []
        for solution, qpu_time in results:
            solutions.append(solution)
            total_q_time += qpu_time

        return self.__aggregate_solutions(solutions, qubo), total_q_time

    def sample_qubo(self, qubo: QUBO) -> Tuple[QUBO, float]:
        if PROFILE:
            profiler = cProfile.Profile()
            profiler.enable() 
        
        res = asyncio.run(self.async_sample_qubo(qubo))

        if PROFILE:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats(pstats.SortKey.TIME).print_stats(5)

        return res

    def __aggregate_solutions(self, solutions: List[QUBO], qubo: QUBO) -> QUBO:
        # Aggregate upper-left qubo with lower-right
        starting_sols = self.__combine_ul_lr(solutions[0], solutions[2])
        # Set missing columns in upper-right qubo to NaN
        ur_qubo_filled = solutions[1].solutions.reindex(columns=starting_sols.columns, fill_value=np.nan)
        # Search the closest assignments between upper-right qubo and merged solution (UL and LR qubos)
        closest_df = self.__get_closest_assignments(starting_sols, ur_qubo_filled)
        # Combine
        combined_df = starting_sols.where(starting_sols == closest_df, np.nan)
        rows_with_nan = starting_sols.isna().any(axis=1) | closest_df.isna().any(axis=1)
        combined_df['energy'] = np.where(rows_with_nan | (starting_sols['energy'].isna() & closest_df['energy'].isna()), 
            np.nan, np.where(starting_sols['energy'].isna(),
                             closest_df['energy'], np.where(closest_df['energy'].isna(), 
                                                            starting_sols['energy'], 
                                                            starting_sols['energy'] + closest_df['energy'])))

        # Conflicts resolution
        qubo.solutions = self.__local_search(combined_df, qubo)
        qubo.solutions = qubo.solutions.reset_index(drop=True).drop_duplicates().nsmallest(n=10, columns='energy')

        return qubo

    def __get_closest_assignments(self, starting_sols: pd.DataFrame, ur_qubo_filled: pd.DataFrame) -> pd.DataFrame:
        def nan_hamming_matrix(df1: pd.DataFrame, df2: pd.DataFrame) -> np.ndarray:
            df1_expanded = np.repeat(df1.values[:, np.newaxis, :], df2.shape[0], axis=1)
            df2_expanded = np.repeat(df2.values[np.newaxis, :, :], df1.shape[0], axis=0)
            mask = ~np.isnan(df1_expanded) & ~np.isnan(df2_expanded)
            mismatches = (df1_expanded != df2_expanded) & mask
            distances = mismatches.sum(axis=2)
            return distances

        distances_matrix = nan_hamming_matrix(starting_sols, ur_qubo_filled)
        closest_indices = np.argmin(distances_matrix, axis=1)
        closest_rows = ur_qubo_filled.iloc[closest_indices].reset_index(drop=True)
        rows_with_nan = closest_rows.isna().any(axis=1)
        closest_rows.loc[rows_with_nan, 'energy'] = np.nan

        return closest_rows

    def __local_search(self, df: pd.DataFrame, qubo: QUBO) -> pd.DataFrame:
        df_no_energy = df.drop(columns=['energy'])
        df['energy'] = df_no_energy.apply(lambda row: row.T @ qubo.qubo_matrix @ row 
                                          if not row.isna().any() else np.nan, axis=1)

        rows_with_nans = df_no_energy[df_no_energy.isna()]
        for idx, row in rows_with_nans.iterrows():
            nan_indices = row.index
            qubo_nans = {
                (i, j): qubo.qubo_dict.get((i, j), 0)
                for i in nan_indices
                for j in nan_indices
            }
            all_zero = all(value == 0 for value in qubo_nans.values())
            
            if all_zero:
                df.loc[idx, nan_indices] = 0
                updated_row = df_no_energy.loc[idx].fillna(0)[:qubo.qubo_matrix.shape[0]]
                df.loc[idx, 'energy'] = updated_row.T @ qubo.qubo_matrix @ updated_row
            else:
                nans_sol = self.__get_result(qubo_nans, qubo.offset, 3, use_default=True)[0].iloc[0]
                
                df.loc[idx, nan_indices] = nans_sol.drop('energy')
                if np.isnan(df.loc[idx, 'energy']):
                    df.loc[idx, 'energy'] = 0
                df.loc[idx, 'energy'] += nans_sol['energy']

        return df

    def __combine_ul_lr(self, ul: QUBO, lr: QUBO) -> pd.DataFrame:
        all_indices = sorted(list(set(ul.rows_idx).union(lr.cols_idx)))
        ul.solutions['tmp'] = 1
        lr.solutions['tmp'] = 1
        merge = pd.merge(ul.solutions, lr.solutions, on='tmp')
        merge['energy'] = merge['energy_x'] + merge['energy_y']
        merge = merge.drop(['energy_x', 'energy_y', 'tmp'], axis=1)
        ul.solutions.drop('tmp', axis=1, inplace=True)
        lr.solutions.drop('tmp', axis=1, inplace=True)
        return merge.reindex(columns=all_indices + ['energy'], fill_value=np.nan)

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
