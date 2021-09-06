import math

import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def exponentiated_quadratic(x1, x2):
    distance = math.exp(-((x1 - x2) / 10) ** 2)
    return distance

def uncorrelated(x1, x2):
    return 1 if x1 == x2 else 0

num_samples = 100
number_of_functions = 100
x = np.linspace(0, num_samples, num_samples, endpoint=False)


mu_0 = np.zeros(num_samples)
x_training = np.array([10, 50, 80])
y_training = np.array([1.5, -0.5, -2])
rest = np.linspace(0, num_samples, num_samples, endpoint=False)

def covariance(x1, x2):
    # print(f"{x1}")
    cov = np.ones(shape=(len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            cov[i][j] = exponentiated_quadratic(x1[i], x2[j])
    return np.asmatrix(cov)

sigma_22 = covariance(x_training, x_training)
sigma_11 = covariance(rest, rest)
sigma_21 = covariance(x_training, rest)
sigma_12 = covariance(rest, x_training)
sigma_22_inverse = inv(sigma_22)

mu_1 = np.matmul(np.matmul(sigma_12, sigma_22_inverse), y_training)
cov_1 = sigma_11 - np.matmul(np.matmul(sigma_12, sigma_22_inverse), sigma_21)

mu_1 = np.asarray(mu_1)
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

def show_unclamped(number_of_functions, number_of_data_points, figure):
    cov =np.ones((number_of_data_points, number_of_data_points))
    for i in range(number_of_data_points):
        for j in range(number_of_data_points):
            cov[i][j] = exponentiated_quadratic(i, j)
    for i in range(number_of_functions):
        samples = multivariate_normal.rvs(mean=np.zeros(number_of_data_points), cov=cov, size=1)
        figure.plot(range(number_of_data_points), samples)

def gaussian_processes_demo(number_of_functions, number_of_data_points, figure):
    pass

for i in range(number_of_functions):
    samples = multivariate_normal.rvs(mean=mu_1[0], cov=cov_1, size=1)
    ax2.plot(x, samples)

show_unclamped(5, 5, ax1)
show_unclamped(10, 100, ax3)

ax2.plot(x_training, y_training, marker="o", markersize=10, markerfacecolor="black", linestyle="None")
for x,y in zip(x_training,y_training):
    label = f"({x},{y})"
    ax2.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.show()
