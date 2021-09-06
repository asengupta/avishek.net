import math

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal


def exponentiated_quadratic(x1, x2):
    distance = math.exp(-((x1 - x2) / 10) ** 2)
    return distance


def uncorrelated(x1, x2):
    return 1 if x1 == x2 else 0


def covariance(x1, x2):
    cov = np.ones(shape=(len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            cov[i][j] = exponentiated_quadratic(x1[i], x2[j])
    return np.asmatrix(cov)


def show_unclamped(number_of_functions, number_of_data_points, kernel, figure, label=False):
    cov = np.ones((number_of_data_points, number_of_data_points))
    for i in range(number_of_data_points):
        for j in range(number_of_data_points):
            cov[i][j] = kernel(i, j)
    for i in range(number_of_functions):
        samples = multivariate_normal.rvs(mean=np.zeros(number_of_data_points), cov=cov, size=1)
        if label:
            figure.plot(range(number_of_data_points), samples, marker="s", markersize=10)
        else:
            figure.plot(range(number_of_data_points), samples)



def gaussian_processes_demo(number_of_functions, number_of_data_points, figure):
    x = np.linspace(0, number_of_data_points, number_of_data_points, endpoint=False)
    x_training = np.array([10, 50, 80])
    y_training = np.array([1.5, -0.5, -2])
    rest = np.linspace(0, number_of_data_points, number_of_data_points, endpoint=False)
    sigma_22 = covariance(x_training, x_training)
    sigma_11 = covariance(rest, rest)
    sigma_21 = covariance(x_training, rest)
    sigma_12 = covariance(rest, x_training)
    sigma_22_inverse = inv(sigma_22)

    mu_1 = np.matmul(np.matmul(sigma_12, sigma_22_inverse), y_training)
    cov_1 = sigma_11 - np.matmul(np.matmul(sigma_12, sigma_22_inverse), sigma_21)

    mu_1 = np.asarray(mu_1)

    for i in range(number_of_functions):
        samples = multivariate_normal.rvs(mean=mu_1[0], cov=cov_1, size=1)
        figure.plot(x, samples)

    figure.plot(x_training, y_training, marker="o", markersize=10, markerfacecolor="black", linestyle="None")
    for x, y in zip(x_training, y_training):
        label = f"({x},{y})"
        figure.annotate(label,  # this is the text
                        (x, y),  # these are the coordinates to position the label
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 10),  # distance from text to points (x,y)
                        ha='center')  # horizontal alignment can be left, right or center


plt.figure()
show_unclamped(5, 5, uncorrelated, plt, label=True)
plt.show()
plt.figure()
show_unclamped(5, 5, exponentiated_quadratic, plt, label=True)
plt.show()
plt.figure()
show_unclamped(10, 100, exponentiated_quadratic, plt)
plt.show()
plt.figure()
gaussian_processes_demo(100, 100, plt)
plt.show()
