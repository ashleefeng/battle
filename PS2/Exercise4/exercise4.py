import sys
import os
import argparse
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

"""
How to run:

python exercise4.py --train train_expr.txt --test test_expr.txt --Kmin 2 --Kmax 10 --output kmeans_output.txt

"""

"""
The function should take the following parameters:
	data - n x d numpy matrix. n = number of samples, d = number of features
	k - int. number of clusters.
	centers - k x d-length numpy matrix. initial value of cluster centers.
	maxit- int. maximum number of iterations.

The myKmeans() function should return a tuple of two items (in this order):
	clusters - n-length numpy array containing the cluster assignment label of each
	sample.
	centers - k x d numpy matrix containing the final k d-dimensional cluster centroids.
"""
def myKmeans(data, k, centers, maxit):

	n = data.shape[0]
	d = data.shape[1]

	clusters = np.zeros((n,), dtype=np.int16)

	iter_num = 0
	
	while iter_num < maxit:

		prev_clusters = np.array(clusters)
	
		# update assignments
		clusters = np.zeros((n,), dtype=np.int16)

		for i in range(n):

			sample = data[i]
			min_dist = 1e6
			min_cluster = 0

			for j in range(k):
				dist = np.linalg.norm(sample - centers[j])
				if dist < min_dist:
					min_dist = dist
					min_cluster = j

			clusters[i] = min_cluster

		# update cluster centers
		cluster_sums = np.zeros((k, d))
		cluster_sizes = np.zeros((k, ))

		for i in range(n):
			assignment = clusters[i]
			cluster_sums[assignment] += data[i]
			cluster_sizes[assignment] += 1
		
		for i in range(k):
			cluster_size = float(cluster_sizes[i])
			if cluster_size != 0:
				centers[i] = cluster_sums[i] / cluster_size

		# print(clusters)
		# print(prev_clusters)
		# print(np.linalg.norm(clusters - prev_clusters))

		if np.linalg.norm(clusters - prev_clusters) == 0:
			print("Converged at iteration %d" %iter_num)
			break

		iter_num += 1

	return (clusters, centers)

"""
The predict() function should take the following
inputs:
	model - the tuple returned by the myKmeans() function.
	test data - a q x d numpy matrix where q = number of query samples.
The predict() function should return the following output:
	clusters - A q-length numpy array containing the cluster assignment of each sample.
"""
def predict(model, test_data):

	q = test_data.shape[0]
	centers = model[1]
	k = len(centers)
	clusters = np.zeros((q, ), dtype=np.int16)

	for i in range(q):
		sample = test_data[i]
		min_dist = 1e6
		min_cluster = 0

		for j in range(k):
			dist = np.linalg.norm(sample - centers[j])
			if dist < min_dist:
				min_dist = dist
				min_cluster = j

		clusters[i] = min_cluster
	
	return clusters

def bic(data, model):

	centers = model[1]
	clusters = model[0]

	n = data.shape[0]
	d = data.shape[1]
	k = len(centers)

	# compute stdev for each feature
	sigmas = np.zeros((d,))
	for i in range(d):
		col = data[:, i]
		sigmas[i] = np.std(col, ddof=1)
	# print sigmas

	m = k * d

	log_likelihood = 0
	for i in range(n):
		assignment = clusters[i]
		for j in range(d):
			xij = data[i, j]
			sigma = sigmas[j]
			# factor = 1/(np.sqrt(2 * np.pi) * sigma)
			# exponent = - (np.linalg.norm(xi - centers[assignment]))**2 / (2 * sigma**2)

			# assume sigma = 1
			factor = 1/(np.sqrt(2 * np.pi) * sigma)
			exponent = - (xij - centers[assignment, j])**2 / (2 * sigma**2)
			log_likelihood += np.log(factor) + exponent


	bic_score = -2 * log_likelihood + m * np.log(n)

	# print(log_likelihood)

	return bic_score

parser = argparse.ArgumentParser()

parser.add_argument("--train")
parser.add_argument("--test")
parser.add_argument("--Kmin")
parser.add_argument("--Kmax")
parser.add_argument("--output")

args = parser.parse_args()

train_filename = args.train
test_filename = args.test
kmin = int(args.Kmin)
kmax = int(args.Kmax)
output_filename = args.output

data = pd.read_csv(train_filename, delimiter='\t', index_col='geneid')
data = data.T
test_data = pd.read_csv(test_filename, delimiter='\t', index_col='geneid')
test_data = test_data.T

best_k = 0
best_bic = None

n = len(data)
d = data.shape[1]

for k in range(kmin, kmax+1):

	print("K = %d" %k)
	# initialize centers with first k samples
	init_centers = np.zeros((k, d))
	for i in range(k):
		init_centers[i] = data.iloc[i]

	model = myKmeans(data.values, k, init_centers, 10)

	bic_score = bic(data.values, model)

	if not best_bic:
		best_k	= k
		best_bic = bic_score
	
	elif bic_score < best_bic:
		best_k = k
		best_bic = bic_score

	print("BIC score = %f\n" %bic_score)

print("Best k = " + str(best_k))
# predicted_clusters = predict(model, test_data)

init_centers = np.zeros((best_k, d))
for i in range(best_k):
	init_centers[i] = data.iloc[i]

model = myKmeans(data.values, best_k, init_centers, 10)
clusters = model[0]
centers = model[1]

pca = PCA(n_components=2)

pca.fit(data)

components = pca.components_
explained_var = pca.explained_variance_ratio_

projection = data.as_matrix().dot(components.T)

train_labels = pd.read_csv("train_label.txt", delimiter='\t', index_col="sample_id")

plt.figure()

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff9900', '#4B0082', '#996633']
markers = ['o', '^', 'X', 's', 'D', '*', 'h']

for i in range(n):
	cluster_assigned = clusters[i]
	true_label = train_labels.iloc[i, 0]
	plt.scatter(projection[i, 0], projection[i, 1], c=colors[cluster_assigned], s=12, marker=markers[true_label], alpha=0.5)


plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.savefig('ex4_train.png', dpi=200)
plt.close()


print("Predicting test data")
test_clusters = predict(model, test_data.as_matrix())

output = pd.DataFrame(data=test_clusters, index=test_data.index, columns=['cluster'])
output.to_csv(output_filename, sep='\t')
print("Results saved as " + output_filename)


# plot test data results
pca = PCA(n_components=2)

pca.fit(test_data)

components = pca.components_
explained_var = pca.explained_variance_ratio_

projection = test_data.as_matrix().dot(components.T)

test_labels = pd.read_csv("test_label.txt", delimiter='\t', index_col="sample_id")

plt.figure()

for i in range(len(test_data)):
	cluster_assigned = test_clusters[i]
	true_label = test_labels.iloc[i, 0]
	plt.scatter(projection[i, 0], projection[i, 1], c=colors[cluster_assigned], s=12, marker=markers[true_label], alpha=0.5)

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.savefig('ex4_test.png', dpi=200)
plt.close()








