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
python exercise4.py --train train_expr.txt --test test_expr.txt --Kmin 2 --Kmax 10 --output kmeans_output.txt

"""

"""
The function should take the following parameters:
• data - n x d numpy matrix. n = number of samples, d = number of features
• k - int. number of clusters.
• centers - k x d-length numpy matrix. initial value of cluster centers.
• maxit- int. maximum number of iterations.
The myKmeans() function should return a tuple of two items (in this order):
• clusters - n-length numpy array containing the cluster assignment label of each
sample.
• centers - k x d numpy matrix containing the final k d-dimensional cluster centroids.
"""
def myKmeans(data, k, centers, maxit):

	n = data.shape[0]
	d = data.shape[1]

	clusters = np.zeros((n,), dtype=np.int16)
	prev_clusters = np.zeros((n,))

	iter_num = 0
	
	while iter_num < maxit:

		prev_clusters = np.array(clusters)
		
		# find cluster centers
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
• model - the tuple returned by the myKmeans() function.
• test data - a q x d numpy matrix where q = number of query samples.
The predict() function should return the following output:
• clusters - A q-length numpy array containing the cluster assignment of each sample.
"""
def predict(model, test_data):

	q = test_data.shape[0]
	centers = model[1]
	k = len(centers)
	clusters = np.zeros((q, ))

	for i in range(q):
		sample = data.iloc[i]
		min_dist = 1e6
		min_cluster = 0

		for j in range(k):
			dist = np.linalg.norm(sample - centers[j])
			if dist < min_dist:
				min_dist = dist
				min_cluster = j

		clusters[i] = min_cluster
	
	return clusters

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
test_data = pd.read_csv(test_filename, delimiter='\t', index_col='geneid')

for k in range(kmin, kmax):

	print("\nk = %d\n" %k)

	n = len(data)
	d = data.shape[1]

	# initialize
	centers = np.zeros((k, d))
	clusters = np.random.choice(k, size=(n,))
	
	# compute initial cluster centers
	cluster_sums = np.zeros((k, d))
	cluster_sizes = np.zeros((k, ))

	for i in range(n):
		assignment = clusters[i]
		cluster_sums[assignment] += data.iloc[i]
		cluster_sizes[assignment] += 1

	for i in range(k):
		cluster_size = float(cluster_sizes[i])
		if cluster_size != 0:
			centers[i] = cluster_sums[i] / cluster_size


	model = myKmeans(data.values, k, centers, 10)

	predicted_clusters = predict(model, test_data)