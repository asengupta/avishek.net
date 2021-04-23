import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines

ax = plt.axes()


# Ensure that the next plot doesn't overwrite the first plot
point  = np.array([0, 0, 0])
normal = np.array([1, 0, 0])

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)

# create x,y
# zz, yy = np.meshgrid(range(-2, 2, 0.5), range(-2, 2, 1))
zz, yy = np.meshgrid(np.linspace(-2., 2., num=5), np.linspace(-2., 2., num=5))
xx = yy*0
# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

# plot the surface
plt3d = plt.subplot(projection='3d')
plt3d.plot_wireframe(0, yy, zz)
plt3d.quiver(0, 0, 0, 1., 0, 0., colors = 'black', linewidths=2, arrow_length_ratio=0.2)
plt3d.text(1, 0, 0, "U=Vector $(1,1,0)$")
plt3d.text(0, -2, 2.5, "$U^{\perp}$=The Plane $x=0$", (1,1,1.5))
# plt3d.text(5, 5, 5, "(5,5,5)")
plt3d.set_xticks(range(-2, 3))
plt3d.set_yticks(range(-2, 3))
plt3d.set_zticks(range(-2, 3))
plt3d.set_xlim([-2,2])
plt3d.set_ylim([-2,2])
plt3d.set_zlim([-2,2])
plt3d.set_xlabel('X axis')
plt3d.set_ylabel('Y axis')
plt3d.set_zlabel('Z axis')

plt.show()
