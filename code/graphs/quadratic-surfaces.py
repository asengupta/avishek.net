import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines

def parabola_level_sets(x,y):
    return x ** 2 - y

def no_cross_term_all_negative(x, y):
    return -x ** 2 - y ** 2

def no_cross_term_both_positive(x, y):
    return x ** 2 + y ** 2

def positive_cross_term(x, y):
    return x ** 2 + y ** 2 + 2 * x * y

def no_cross_term_saddle(x, y):
    return x ** 2 - y ** 2

def positive_cross_term_saddle(x, y):
    return x ** 2 - y ** 2 - 2*x*y


def plot_quadratic(f, fig):
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array(f(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1,x_2)$')
    ax.text2D(1, 1, "2D Text")

fig = plt.figure()
# plot_quadratic(no_cross_term, fig)
# plot_quadratic(positive_cross_term, fig)
# plot_quadratic(no_cross_term_saddle, fig)
# plot_quadratic(positive_cross_term_saddle, fig)
# plot_quadratic(no_cross_term_both_positive, fig)
plot_quadratic(parabola_level_sets, fig)
plt.show()
