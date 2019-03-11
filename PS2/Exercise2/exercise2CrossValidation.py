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
python3 exercise2CrossValidation.py --features trainFeatures.txt
                                          --labels trainLabel.txt
                                           --kernel linear
                                           --output linear_model_cv_accuracy.txt
python3 exercise2CrossValidation.py --features trainFeatures.txt
                                          --labels trainLabel.txt
                                           --kernel gaussian
                                           --output gaussian_model_cv_accuracy.txt
python3 exercise2CrossValidation.py --features trainFeatures.txt
                                          --labels trainLabel.txt
                                           --kernel polynomial_3
                                           --output polynomial_3_model_cv_accuracy.txt
python3 exercise2CrossValidation.py --features trainFeatures.txt
                                          --labels trainLabel.txt
                                           --kernel polynomial_6
                                           --output polynomial_6_model_cv_accuracy.txt
"""

def standardize(X):

	n_cols = X.shape[1]
	n_rows = X.shape[0]

	for i in range(n_cols):
		col = X[:, i]
		std = np.std(col, ddof=1)
		avg = np.mean(col)
		X[:, i] = (col - avg) / std

	return X


def train(X, y, clf):
	clf.fit(X, y)
	return clf

def test(testX, testY, clf):
	prediction = clf.predict(testX)
	n_test = len(testY)
	n_correct = 0
	for i in range(n_test):
		if prediction[i] == testY[i]:
			n_correct += 1

	return float(n_correct)/n_test

def svcCrossValidation(X, y, k, kernel):
	n_samples = len(X)
	n_subset = len(X)/k
	accuracy_sum = 0

	# generate indices of training and test set for cross validatoin
	for i in range(k):
		if i == 0:
			train_indices = range(n_subset, n_samples)
			test_indices = range(0, n_subset)
		elif i == k-1:
			train_indices = range(0, (i * n_subset))
			test_indices = range((i * n_subset), n_samples)
		else:
			train_indices = range(0, (i * n_subset)) + range((i + 1) * n_subset, n_samples)
			test_indices = range((i * n_subset), ((i+1) * n_subset))

		trainX = X[train_indices]
		trainY = y[train_indices]

		testX = X[test_indices]
		testY = y[test_indices]

		# create clf

		kernel_type = kernel[0]
		degree = kernel[1]

		clf = SVC(kernel=kernel_type, degree=degree)

		clf = train(trainX, trainY, clf)

		accuracy = test(testX, testY, clf)
		print(accuracy)

		accuracy_sum += accuracy

	return float(accuracy_sum)/k

parser = argparse.ArgumentParser(description="Use SVM to predict splicing.")

parser.add_argument("--features")
parser.add_argument("--labels")
parser.add_argument("--kernel")
parser.add_argument("--output")

args = parser.parse_args()

features_filename = args.features
labels_filename = args.labels
output_filename = args.output

kernel_tranlate = {"linear": ["linear", 0], "gaussian": ["rbf", 0], "polynomial_3": ["poly", 3], "polynomial_6": ["poly", 6]}

kernel = kernel_tranlate[args.kernel]

X = np.loadtxt(features_filename, delimiter='\t')
X = standardize(X)

n_samples = len(X)

y = np.zeros((n_samples,))
labels_file = open(labels_filename, 'r')

line_number = 0
for line in labels_file:
	tokens = line.rstrip('\n').split(' ')
	
	is_spliced = tokens[2]
	if is_spliced == 'S':
		y[line_number] = 1
	
	line_number += 1

labels_file.close()
# print(y[:20])

cv_score = svcCrossValidation(X, y, 5, kernel)
print("average: " + str(cv_score))

output_file = open(output_filename, 'w')
output_file.write(str(cv_score) + '\n')
output_file.close()





















