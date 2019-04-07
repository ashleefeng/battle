import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

filename = "expr_ceph_utah_1000.txt"

data = pd.read_csv(filename, delimiter='\t', index_col=0)
n_rows = data.shape[0]
n_cols = data.shape[1]
corr_matrix = np.zeros((n_cols, n_cols))

for i in range(n_cols):
	for j in range(i, n_cols):
		corr_coef = abs(pearsonr(data.iloc[:, i], data.iloc[:, j])[0])
		corr_matrix[i, j] = corr_coef
		corr_matrix[j, i] = corr_coef

np.savetxt("corr_matrix.csv", corr_matrix, delimiter=',')

