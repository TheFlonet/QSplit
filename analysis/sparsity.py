import pandas as pd
import json
import numpy as np

def main():
    df = pd.read_csv('../dataset/problems.csv')
    problems = list(df.itertuples(index=False, name=None))

    results = []

    for _, (problem_name, _) in enumerate(problems):
        with open(f'../dataset/qubo_instances/{problem_name}.json') as json_file:
            r = json.load(json_file)
            arr = np.array(r['qubo_matrix'])
            sparsity = 100 * np.count_nonzero(arr) / (arr.shape[0] * (arr.shape[0] + 1) / 2)
            results.append((problem_name, np.round(sparsity, 2)))

    results_df = pd.DataFrame(results, columns=['problem_name', 'sparsity_percentage'])
    results_df.to_csv('../dataset/problems_sparsity.csv', index=False)


if __name__ == '__main__':
    main()