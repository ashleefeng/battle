import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.covariance import GraphicalLasso

cov_matrix = np.loadtxt("cov_matrix.csv", delimiter=',')

model = GraphicalLasso(alpha=0.55)
model.fit(cov_matrix)
prec_matrix = model.get_precision()
# print(prec_matrix)
# np.savetxt("precision_matrix.csv", prec_matrix, delimiter=',')
n = cov_matrix.shape[0]
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