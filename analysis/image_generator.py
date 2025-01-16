import numpy as np
from PIL import Image
import pandas as pd
import json

def save(mat, output_path):
    mat = np.array(mat, dtype=float)
    img = np.zeros_like(mat, dtype=np.uint8)  
    img[mat != 0] = 255
    
    altezza, larghezza = mat.shape
    rgb_img = np.zeros((altezza, larghezza, 3), dtype=np.uint8)
    rgb_img[..., 0] = img  # R
    rgb_img[..., 1] = img  # G
    rgb_img[..., 2] = img  # B
    
    immagine = Image.fromarray(rgb_img)
    immagine.save(output_path)

def main():
    df = pd.read_csv('../dataset/problems.csv')
    problems = list(df.itertuples(index=False, name=None))

    for _, (problem_name, _) in enumerate(problems):
        with open(f'../dataset/qubo_instances/{problem_name}.json') as json_file:
            r = json.load(json_file)
            save(r['qubo_matrix'], f'./dataset/img/{problem_name}.png')

if __name__=='__main__':
    main()
