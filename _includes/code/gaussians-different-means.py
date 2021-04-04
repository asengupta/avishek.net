import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

# mu = 0
mus = [0, 1, 2, 3]
variance = 1
sigma = math.sqrt(variance)
plt.grid()
for mu in mus:
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.text(mu, 0.4, f'$\mu$={mu}')
plt.show()
