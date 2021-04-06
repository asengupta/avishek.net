import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

# mu = 0
mu = 0
variances = [1, 2, 3, 4]
plt.grid()
for variance in variances:
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.text(mu, stats.norm(mu, sigma).pdf(mu), f'$\sigma$={variance}')
plt.show()
