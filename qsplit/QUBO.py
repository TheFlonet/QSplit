from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.linalg import lu
from util import QUBO_DICT


class QUBO:
    def __init__(self, qubo_dict: QUBO_DICT, cols_idx: List[int] | None = None, 
                 rows_idx: List[int] | None = None, to_transform: bool = True):
        self.qubo_dict: QUBO_DICT = qubo_dict
        if cols_idx is None and rows_idx is None:
            raise ValueError("QUBO class requires at least one of cols_idx or rows_idx to be not None")
        self.cols_idx: List[int] = cols_idx if cols_idx is not None else rows_idx
        self.rows_idx: List[int] = rows_idx if rows_idx is not None else cols_idx
        self.solutions: pd.DataFrame | None = None
        self.qubo_matrix: np.ndarray | None = None
        self.__to_transform: bool = to_transform
        self.__from_dict_to_matrix((len(self.rows_idx), len(self.cols_idx)))

    def __is_upper_triangular(self) -> bool:
        if self.qubo_matrix is None:
            return False

        if not np.allclose(self.qubo_matrix, np.triu(self.qubo_matrix)):
            return False

        return True

    def __from_dict_to_matrix(self, dims: Tuple[int, int]) -> None:
        self.qubo_matrix = np.zeros(dims)
        for k, v in self.qubo_dict.items():
            self.qubo_matrix[k[0] % dims[0], k[1] % dims[1]] = v
        if not self.__to_transform:
            return 
        if self.__is_upper_triangular():
            return
        
        self.qubo_matrix = lu(self.qubo_matrix, permute_l=True)[1]
        self.__from_matrix_to_dict()

    def __from_matrix_to_dict(self) -> None:
        self.qubo_dict = {}
        for i in range(self.qubo_matrix.shape[0]):
            for j in range(self.qubo_matrix.shape[1]):
                if self.qubo_matrix[i, j] != 0:
                    self.qubo_dict[(i + min(self.rows_idx), j + min(self.cols_idx))] = float(self.qubo_matrix[i, j])
