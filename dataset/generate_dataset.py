import os
import csv
import dimod

def main():
    lp_files = [f for f in os.listdir('.') if f.endswith('.lp')]
    problems = sorted([(p, len(dimod.lp.load(p).variables)) for p in lp_files], 
                      key=lambda x: x[1])

    with open('problems.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["problem name", "#variables"])
        writer.writerows(problems)


if __name__=='__main__':
    main()