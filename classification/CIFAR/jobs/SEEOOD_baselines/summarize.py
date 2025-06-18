import os
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="details")
parser.add_argument('--experiment', type=str)
args = parser.parse_args()

EXP_DSET = args.experiment

index = -2
file_path = os.path.join('jobs', 'out', f'vos-{EXP_DSET}.log')
with open(file_path, 'r') as f:
        lines = f.readlines()
        # print(lines[index])
        results = lines[index].strip().split(" & ")
        auc, tpr95, tpr99 = float(results[2]), float(results[0]), float(results[1])


print(f"Summary for {EXP_DSET}")
print(f"AUC: {auc:.4f}\nTPR95: {tpr95:.4f}\nTPR99: {tpr99:.4f}\n\n")
