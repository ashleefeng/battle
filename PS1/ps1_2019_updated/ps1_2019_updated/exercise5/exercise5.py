import argparse
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing

print("\nPS1 Exercise 5")

parser = argparse.ArgumentParser(description="Build a logistic regression model to predict group 0 and group 1 breast cancer patients.")

parser.add_argument("--X", help="training set features")
parser.add_argument("--Y", help="training set outcomes")
parser.add_argument("--testX", help="test set features")
parser.add_argument("--testY", help="test set outcomes")

args = parser.parse_args()


X_filename = args.X
Y_filename = args.Y
testX_filename = args.testX
testY_filename = args.testY

X = pd.read_csv(X_filename, index_col=0).to_numpy().T
Y = np.ravel(pd.read_csv(Y_filename, index_col=0).to_numpy())
testX = pd.read_csv(testX_filename, index_col=0).to_numpy().T
testY = np.ravel(pd.read_csv(testY_filename, index_col=0).to_numpy())

X = preprocessing.scale(X)
testX = preprocessing.scale(testX)

clf = LogisticRegression(solver='lbfgs')
clf.fit(X, Y)
y_pred = clf.predict(testX)

print("\nTraining using all genes")
print("precision: " + str(precision_score(testY, y_pred)))
print("recall: " + str(recall_score(testY, y_pred)))
# print("accuracy: " + str(clf.score(testX, testY)))

print("\nTraining using only the first 10 genes")
X10 = X[:, :10]
testX10 = testX[:, :10]

clf2 = LogisticRegression(solver='lbfgs')
clf2.fit(X10, Y)
y_pred = clf2.predict(testX10)

print("precision: " + str(precision_score(testY, y_pred)))
print("recall: " + str(recall_score(testY, y_pred)))
# print("accuracy: " + str(clf2.score(testX10, testY)))