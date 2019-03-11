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
python ex3-2.py --input log_transformed_and_standardized_expr.txt --output ex3-2.txt

"""

parser = argparse.ArgumentParser()

parser.add_argument("--input")
parser.add_argument("--output")

args = parser.parse_args()

input_filename = args.input
output_filename = args.output

data = pd.read_csv(input_filename, delimiter='\t', index_col='EnsemblID')

data = data.T 

pca = PCA(n_components=10)

pca.fit(data)

components = pca.components_
explained_var = pca.explained_variance_ratio_

output_file = open(output_filename, 'w')

for v in explained_var:
	output_file.write("%.3f%%\n" %(v*100))

output_file.close()

