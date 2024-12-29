import os
import dimod
import shutil

def main():
    lp_files = [f for f in os.listdir('./all_problems') if f.endswith('.lp')]
    
    num = 0
    for p_path in lp_files:
        prev = num
        try:
            problem = dimod.lp.load(p_path)
            original_var_num = len(problem.variables)
            q_problem = dimod.cqm_to_bqm(problem)[0].to_qubo()[0]
            ids = set()
            for i, j in q_problem.keys():
                ids.add(i)
                ids.add(j)
            if original_var_num != len(ids):
                print('Problem with variables: ', p_path)
                num += 1
        except Exception as e:
            num += 1
            print('Error with', p_path, e)
        if num == prev:
            shutil.copy2(p_path, '../')


if __name__=='__main__':
    main()