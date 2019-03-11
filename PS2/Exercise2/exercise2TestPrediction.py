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
python3 exercise2TestPrediction.py --features_train trainFeatures.txt
                                         --labels_train trainLabel.txt
                                         --features_test testFeatures.txt
                                         --labels_test testLabel.txt
                                           --kernel gaussian
                                           --output test_accuracy.txt
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

parser = argparse.ArgumentParser(description="Use SVM to predict splicing.")

parser.add_argument("--features_train")
parser.add_argument("--labels_train")
parser.add_argument("--features_test")
parser.add_argument("--labels_test")
parser.add_argument("--kernel")
parser.add_argument("--output")

args = parser.parse_args()

features_train_filename = args.features_train
labels_train_filename = args.labels_train

features_test_filename = args.features_test
labels_test_filename = args.labels_test

output_filename = args.output

kernel_tranlate = {"linear": ["linear", 0], "gaussian": ["rbf", 0], "polynomial_3": ["poly", 3], "polynomial_6": ["poly", 6]}

kernel = kernel_tranlate[args.kernel]

# Load training data

trainX = np.loadtxt(features_train_filename, delimiter='\t')
trainX = standardize(trainX)

n_samples = len(trainX)

trainY = np.zeros((n_samples,))
labels_file = open(labels_train_filename, 'r')

line_number = 0
for line in labels_file:
	tokens = line.rstrip('\n').split(' ')
	
	is_spliced = tokens[2]
	if is_spliced == 'S':
		trainY[line_number] = 1
	
	line_number += 1

labels_file.close()

# Load test data

testX = np.loadtxt(features_test_filename, delimiter='\t')
testX = standardize(testX)

n_test = len(testX)

testY = np.zeros((n_test,))
labels_file = open(labels_test_filename, 'r')

line_number = 0
for line in labels_file:
	tokens = line.rstrip('\n').split(' ')
	
	is_spliced = tokens[2]
	if is_spliced == 'S':
		testY[line_number] = 1
	
	line_number += 1

labels_file.close()

# Train clf

kernel_type = kernel[0]
degree = kernel[1]

clf = SVC(kernel=kernel_type, degree=degree)
clf.fit(trainX, trainY)
score = test(testX, testY, clf)
print("Accuracy: " + str(score))

output_file = open(output_filename, 'w')
output_file.write(str(score) + '\n')
output_file.close()