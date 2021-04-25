import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines

ax = plt.axes()


# Ensure that the next plot doesn't overwrite the first plot

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set

# create x,y

# plot the surface
plt3d = plt.subplot(projection='3d')


def plot_surface(fn, color):
    xx, yy = np.meshgrid(np.linspace(-2., 2., num=15), np.linspace(-2., 2., num=15))
    zz = fn(xx,yy)
    plt3d.plot_wireframe(xx, yy, zz, color=color)


plot_surface(lambda xx,yy: -(xx+yy), 'red')
plot_surface(lambda xx,yy: -(xx+2*yy)/3., 'blue')
z=np.linspace(-5,5,10)
x=z
y=-2*z
plt3d.plot(x,y,z, linewidth=4)
plt3d.text(-2,1,1, 'x+y+z=0')
plt3d.text(1,1,0, 'x+2y+3z=0')
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
