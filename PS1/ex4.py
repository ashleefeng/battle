from scipy import stats
import numpy as np

sox2 = [10.9, 7.6, 11.7, 11.8, 12.5, 8.7, 10.8, 10.3, 10.9, 9.3]
tnnt2 = [15.6, 15.1, 19.4, 17.1, 17.2, 15.5, 17.6, 17.6, 14.4, 14.6]
mu1 = np.mean(sox2)
mu2 = np.mean(tnnt2)
sigma1 = np.std(sox2)
sigma2 = np.std(tnnt2)
cov = np.cov([sox2, tnnt2])[0, 1]
phi = cov/(sigma1 * sigma2)

pearson = stats.pearsonr(sox2, tnnt2)

print("pearson correlation: %f" %pearson[0])
print("p-value: %f" %pearson[1])
print("mu1: %f \nmu2: %f" %(mu1, mu2))
print("covariance: %f" %cov)

x1 = 10.2
x2 = mu2 + sigma2 / sigma1 * phi * (x1 - mu1)
# print(phi * sigma2/sigma1 * (x - np.mean(sox2)))
print("Expected TNNT2 expression level: %f" %x2)

