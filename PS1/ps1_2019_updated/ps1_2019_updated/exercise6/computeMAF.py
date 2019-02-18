import argparse
import pandas as pd 
import numpy as np

print("\nPS1 Exercise 6a")

parser = argparse.ArgumentParser(description="Compute Minor Allele Frequency of each SNP.")

parser.add_argument("--X")
parser.add_argument("--output")

args = parser.parse_args()

X_filename = args.X 
output_filename = args.output 

X = pd.read_csv(X_filename, index_col=0)
n_snps = len(X.columns)
out = pd.DataFrame(data=np.zeros(n_snps), index=X.columns)


count = 0
for col in X:
	counts = np.zeros((4,))
	vc = X[col].value_counts()
	keys = vc.keys()
	for k in keys:
		counts[k] = vc[k]
	total = np.sum(counts)
	A_freq = (counts[0] + 0.5 * counts[1]) / total
	B_freq = (counts[2] + 0.5 * counts[1]) / total
	if A_freq > B_freq:
		out.iloc[count, 0] = B_freq
	else:
		out.iloc[count, 0] = A_freq
	count += 1

out.to_csv(output_filename, sep='\t', header=False)

print("Results saved as " + output_filename)

count1 = 0
count2 = 0
count3 = 0

for i in range(n_snps):
	freq = out.iloc[i, 0]
	if freq > 0.03:
		count1 += 1
	if freq > 0.05:
		count2 += 1
	if freq > 0.1:
		count3 += 1

print("%d SNPs have MAF > 0.03" %count1)
print("%d SNPs have MAF > 0.05" %count2)
print("%d SNPs have MAF > 0.1" %count3)