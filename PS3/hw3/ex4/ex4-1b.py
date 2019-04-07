import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats

corr_matrix = np.loadtxt("corr_matrix.csv", delimiter=',')

n = corr_matrix.shape[0]

tau_list = np.arange(0.25, 0.85, 0.05)
rsq_list = []

for tau in tau_list:
	adj_matrix = np.zeros((n, n))
	for i in range(n):
		for j in range(i, n):
			if corr_matrix[i, j] > tau:
				adj_matrix[i, j] = 1
				adj_matrix[j, i] = 1

	degree_list = np.sum(adj_matrix, axis=0) - 1
	# print(degree_list)

	pd = np.zeros((np.int(degree_list.max()+1,)))
	for dg in degree_list:
		pd[int(dg)] += 1.0/n 

	x = []
	y = []
	for i in range(int(degree_list.max()+1)):
		if pd[i] != 0 and i != 0:
			x.append(np.log(i))
			y.append(np.log(pd[i]))
	
	slope, intercept, r, p, stderr = stats.linregress(x, y)
	rsq_list.append(r**2)
	x = np.array(x)
	y = np.array(y)
	plt.figure()
	plt.plot(x, y, 'o', label='tau = %.2f' %tau)
	plt.plot(x, intercept + slope*x, 'r', label="r-squared = %.2f" %r**2)
	plt.xlabel("log(degree)")
	plt.ylabel("log(Pr(degree))")
	plt.legend()
	plt.savefig("tau = %.2f.png" %tau)
	plt.close()

	if abs(tau-0.5) < 1e-2:
		plt.figure()
		plt.hist(degree_list, bins=50)
		plt.xlabel("degree")
		plt.ylabel("count")
		plt.title("tau = 0.5")
		plt.savefig("degree_hist_tau=0.5.png")
		plt.close()


plt.figure()
plt.plot(tau_list, rsq_list, 'o')
plt.xlabel("tau")
plt.ylabel("r-squared")
plt.savefig("R-squared_vs_tau.png")
plt.close()






