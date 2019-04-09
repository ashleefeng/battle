import numpy as np 
import pandas as pd

wgcna = np.loadtxt("wgcna_adj_matrix.csv", delimiter=',')
glasso = np.loadtxt("glasso_adj_matrix.csv", delimiter=',')

n = wgcna.shape[0]

nE_wgcna = 0
nE_glasso = 0
n_both = 0
for i in range(n):
	for j in range(i, n):
		w = wgcna[i, j]
		g = glasso[i, j]

		if w == 1 and g == 1:
			nE_glasso += 1
			nE_wgcna += 1
			n_both += 1
		elif w == 1:
			nE_wgcna += 1
		elif g == 1:
			nE_glasso += 1

print("Total number of edges in wgcna: %d" %nE_wgcna)
print("Total number of edges in glasso: %d" %nE_glasso)
print("Total number of edges in both: %d (%.2f%% of wgcna, %.2f%% of glasso)" %(n_both, float(n_both)/nE_wgcna*100, float(n_both)/nE_glasso*100))

# Total number of edges in wgcna: 8690
# Total number of edges in glasso: 3660
# Total number of edges in both: 3660 (42.12% of wgcna, 100.00% of glasso)