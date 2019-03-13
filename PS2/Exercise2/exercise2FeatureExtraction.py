import argparse
import numpy as np
import pandas as pd

"""

How to run:
python exercise2FeatureExtraction.py --input trainSequence.txt --order featureOrder.txt --output trainFeatures.txt
python exercise2FeatureExtraction.py --input testSequence.txt --order featureOrder.txt --output testFeatures.txt

"""

parser = argparse.ArgumentParser(description="Use SVM to predict splicing.")

parser.add_argument("--input")
parser.add_argument("--order")
parser.add_argument("--output")

args = parser.parse_args()

input_filename = args.input
order_filaname = args.order
output_filename = args.output

order = np.loadtxt(order_filaname, dtype=str)
n_features = len(order)

# count number of rows in input file
n_lines = 0
for line in open(input_filename):
	n_lines += 1

features = pd.DataFrame(data=np.zeros((n_lines, n_features), dtype=np.uint8), index=range(n_lines), columns=order)

input_file = open(input_filename, 'r')

line_number = 0

for line in input_file:

	if line_number % 200 == 0:
		print("Done with %d rows." %line_number)

	seq = line.rstrip('\n')
	n_bp = len(seq)
	for i in range(n_bp):
		# k-mer counting
		for j in [2, 3, 4]:
			if (i + j > n_bp):
				break
			else:
				kmer = seq[i: i+j]
				features.loc[line_number, kmer] += 1
	
	line_number += 1

input_file.close()
features.to_csv(output_filename, sep='\t', header=False, index=False)
