import math

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def exponentiated_quadratic(x1, x2):
    return math.exp(-((x1 - x2)/10)**2)

def uncorrelated(x1, x2):
    return 1 if x1 == x2 else 0

num_samples = 100
number_of_functions = 3
x = np.linspace(0, num_samples, num_samples, endpoint=False)

cov =np.ones((num_samples, num_samples))

for i in range(num_samples):
    for j in range(num_samples):
        cov[i][j] = uncorrelated(i,j)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

for i in range(number_of_functions):
    samples = multivariate_normal.rvs(mean=np.zeros(num_samples), cov=cov, size=1)
    ax1.plot(x, samples)

plt.show()
