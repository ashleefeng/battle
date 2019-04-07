import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.covariance import GraphicalLasso

filename = "expr_ceph_utah_1000.txt"

data = pd.read_csv(filename, delimiter='\t', index_col=0)
n_rows = data.shape[0]
n_cols = data.shape[1]
cov_matrix = np.cov(data.T)
# print(cov_matrix.shape)

model = GraphicalLasso(alpha=0.55)
model.fit(cov_matrix)
prec_matrix = model.get_precision()
# print(prec_matrix)
np.savetxt("precision_matrix.csv", prec_matrix, delimiter=',')
n = n_cols
adj_matrix = np.zeros((n, n))
for i in range(n):
	for j in range(i, n):
		if prec_matrix[i, j] != 0:
			adj_matrix[i, j] = 1
			adj_matrix[j, i] = 1

degree_list = np.sum(adj_matrix, axis=0) - 1
plt.figure()
plt.hist(degree_list)
plt.xlabel("degree")
plt.ylabel("count")
plt.title("Glasso")
plt.savefig("degree_hist_glasso_net.png")
plt.close()

