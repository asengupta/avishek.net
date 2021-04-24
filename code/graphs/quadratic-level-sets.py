import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
from labellines import labelLine, labelLines

def parabola_surface(x,y):
    return x ** 2 - y

def parabola_level_set_for(c):
    def parabola_level_set(x):
        return x ** 2 - c
    return parabola_level_set

def circle_surface(x, y):
    return x * x + y * y

def circle_level_set_for_positive(c):
    def circle_level_set(x):
        return numpy.sqrt(c * c - x * x)
    return circle_level_set

def circle_level_set_for_negative(c):
    def circle_level_set(x):
        return -numpy.sqrt(c * c - x * x)
    return circle_level_set

def plot_quadratic(f, spread, ax):
    # ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(spread[0]*2, spread[1]*2, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array(f(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, alpha=0.3)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1,x_2)$')

def plot_level_set(f, surface_function, c, spread, ax):
    X = np.linspace(spread[0], spread[1], 100)
    Y = f(X)
    Z = np.full_like(Y, c)
    Z = surface_function(X,Y)
    ax.plot3D(X,Y,Z, linewidth=3)


def plot_level_sets(set_range, level_set_generator, surface_function):
    for c in set_range:
        p = level_set_generator(c)
        plot_level_set(p, surface_function, c, quadratic_spread, ax)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
quadratic_spread = (-3, 3)
circle_spread = range(0, 5)

plot_quadratic(circle_surface, quadratic_spread, ax)
plot_level_sets(circle_spread, circle_level_set_for_positive, circle_surface)
plot_level_sets(circle_spread, circle_level_set_for_negative, circle_surface)

# plot_quadratic(parabola_surface, spread, ax)
# plot_level_sets(range(0, 11, 2), parabola_level_set_for)
plt.show()
