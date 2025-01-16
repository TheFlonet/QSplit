from numpy import floating, integer
from typing import Hashable, Mapping, Tuple

QUBO_DICT = Mapping[tuple[Hashable, Hashable], float | floating | integer]

def normalize_qubo(coeffs: QUBO_DICT) -> Tuple[QUBO_DICT, int]:
    ids = set()
    new_coeffs = {}
    for (a, b), v in coeffs.items():
        a = int(a[1:]) - 2
        b = int(b[1:]) - 2
        ids.add(a)
        ids.add(b)
        if a < b:
            new_coeffs[(a, b)] = v
        else:
            new_coeffs[(b, a)] = v
    
    return new_coeffs, len(ids)