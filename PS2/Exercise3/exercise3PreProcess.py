import sys
import os
import argparse
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

"""
python exercise3PreProcess.py --input counts.txt --output log_transformed_and_standardized_expr.txt

"""

parser = argparse.ArgumentParser()

parser.add_argument("--input")
parser.add_argument("--output")

args = parser.parse_args()

input_filename = args.input
output_filename = args.output

data = pd.read_csv(input_filename, delimiter='\t', index_col='EnsemblID')

# log-transform

data = np.log2(data + 1)

# z-score

for i in range(len(data)):
      row = data.iloc[i]
      mean = np.mean(row)
      std = np.std(row, ddof=1)
      data.iloc[i] = (row - mean) / std

data.to_csv(output_filename, sep='\t')

