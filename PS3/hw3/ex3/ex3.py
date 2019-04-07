import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='o', color='w', label='CEU', markerfacecolor='r', markersize=5), Line2D([0], [0], marker='o', color='w', label='YRI', markerfacecolor='g', markersize=5), Line2D([0], [0], marker='o', color='w', label='JPT', markerfacecolor='b', markersize=5), Line2D([0], [0], marker='o', color='w', label='HCB', markerfacecolor='k', markersize=5)]


genotype_filename = "genotype_population.csv"
population_filename = "population_info.csv"

genotype_table = pd.read_csv(genotype_filename, delimiter=',', index_col=0)
population_table = pd.read_csv(population_filename, delimiter=',', index_col=0)


color_seq = []
for i in population_table.index:
	if population_table["V2"][i] == "CEU":
		color_seq.append('r')
	if population_table["V2"][i] == "YRI":
		color_seq.append('g')
	if population_table["V2"][i]== "JPT":
		color_seq.append('b')
	if population_table["V2"][i] == "HCB":
		color_seq.append('k')

# print(genotype_table.head())
# print(population_table.head())

# Standardize each SNP to have 0 mean and 1 stdev

X = np.zeros(genotype_table.shape)

counter = 0

for c in genotype_table:
	column = genotype_table[c]
	mu = np.mean(column)
	sigma = np.std(column, ddof=1)
	X[:, counter] = (column - mu)/sigma
	counter += 1

# print(genotype)

pca = PCA(n_components=20, copy=False)

pca.fit(X)

components = pca.components_
explained_var = pca.explained_variance_ratio_

plt.figure()
plt.bar(range(1, 21), explained_var)
plt.savefig("explained_var1.png")
plt.close()


pca2 = PCA(n_components=20, copy=False)

pca2.fit(X.T)

components2 = pca2.components_
explained_var2 = pca2.explained_variance_ratio_

plt.figure()
plt.bar(range(1, 21), explained_var2)
plt.savefig("explained_var2.png")
plt.close()

X_trans = pca.fit_transform(X)
# print(X_trans.shape)

plt.figure()
plt.scatter(X_trans[:, 0], X_trans[:, 1], s=5, c=color_seq)
plt.title("PC1 VS PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(handles=legend_elements)
plt.savefig("pc1_vs_pc2.png")
plt.close()

plt.figure()
plt.scatter(X_trans[:, 1], X_trans[:, 2], s=5, c=color_seq)
plt.title("PC2 VS PC3")
plt.xlabel("PC2")
plt.ylabel("PC3")
plt.legend(handles=legend_elements)
plt.savefig("pc2_vs_pc3.png")
plt.close()

plt.figure()
plt.scatter(X_trans[:, 2], X_trans[:, 3], s=5, c=color_seq)
plt.title("PC3 VS PC4")
plt.xlabel("PC3")
plt.ylabel("PC4")
plt.legend(handles=legend_elements)
plt.savefig("pc3_vs_pc4.png")
plt.close()

plt.figure()
plt.scatter(X_trans[:, 7], X_trans[:, 8], s=5, c=color_seq)
plt.title("PC8 VS PC9")
plt.xlabel("PC8")
plt.ylabel("PC9")
plt.legend(handles=legend_elements)
plt.savefig("pc8_vs_pc9.png")
plt.close()
