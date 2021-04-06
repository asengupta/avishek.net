import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines

ax = plt.axes()


# Ensure that the next plot doesn't overwrite the first plot
point  = np.array([5, 5, 5])
normal = np.array([-1, -1, 2])

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)

# create x,y
xx, yy = np.meshgrid(range(-10, 10), range(-10, 10))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_wireframe(xx, yy, z, rstride=5, cstride=5)
plt3d.quiver(0, 0, 0, 5., 5., 5., colors = 'black', arrow_length_ratio = 0.2, linewidths=2)
plt3d.quiver(0, 0, 0, -10, -10, 20, colors = 'black', arrow_length_ratio = 0.075, linewidths=2)
plt3d.text(1, 0, -1, "0")
plt3d.text(-10, -10, 20, "(-1,-1,2)")
plt3d.text(5, 5, 5, "(5,5,5)")
plt3d.set_xlabel('X axis')
plt3d.set_ylabel('Y axis')
plt3d.set_zlabel('Z axis')

plt.show()

# ax.scatter(points2[0], point2[1], point2[2], color='green')
# m = -(6/4)
# x = np.linspace(-5,5,10)
# y = m*x
# plt.grid()
#
# plt.axhline(0, color='black')
# plt.axvline(0, color='black')
# plt.xlim(-8,8)
# plt.ylim(-8,8)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title('Graph of 6x+4y=0')
# plt.plot(x, y, '-r', label='6x+4y=0')
# ax.arrow(0.0, 0.0, 6.0, 4.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=2)
# ax.arrow(0.0, 0.0, 2.0, -3.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=5)
# ax.arrow(0.0, 0.0, -4.0, 6.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=5)
# plt.title("A Line and its Normal Vector",fontsize=10)
# plt.text(6, 3, "(6,4)")
# plt.text(3, -4, "(2,-3)")
# plt.text(-3, 6, "(-4,6)")
# plt.text(0.5, -0.5, "$\pi /2^{\circ}$")
# plt.text(-0.5, 1, "$\pi /2^{\circ}$")
