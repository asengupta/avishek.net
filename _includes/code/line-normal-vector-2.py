import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines

ax = plt.axes()

m = -(6/4)
x = np.linspace(-5,5,10)
y = m*x
plt.grid()

plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Graph of 6x+4y=0')
plt.plot(x, y, '-r', label='6x+4y=0')
ax.arrow(0.0, 0.0, 6.0, 4.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=2)
ax.arrow(0.0, 0.0, 2.0, -3.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=5)
ax.arrow(0.0, 0.0, -4.0, 6.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=5)
plt.title("A Line and its Normal Vector",fontsize=10)
plt.text(6, 3, "(6,4)")
plt.text(3, -4, "(2,-3)")
plt.text(-3, 6, "(-4,6)")
plt.text(0.5, -0.5, "$\pi /2^{\circ}$")
plt.text(-0.5, 1, "$\pi /2^{\circ}$")
plt.show()
