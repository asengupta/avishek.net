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
plt.xlim(-8,20)
plt.ylim(-8,40)
plt.gca().set_aspect('equal', adjustable='box')
ax.arrow(0.0, 0.0, 2.0, 3.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=1)
ax.arrow(0.0, 0.0, 4.0, 6.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=1)

ax.arrow(0.0, 0.0, 5.0, 10.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=1)
ax.arrow(0.0, 0.0, 10.0, 20.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=1)
ax.arrow(0.0, 0.0, 15.0, 30.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=1)

ax.arrow(15.0, 30.0, 4.0, 6.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black', lw=1)
ax.arrow(0.0, 0.0, 19.0, 36.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='red', lw=1)

ax.plot([19, 19], [36, 0], '--')
ax.plot([19, 0], [36, 36], '--')
plt.title("Linear Combination of 2 Vectors",fontsize=10)
offset = 1
plt.text(2 + offset, 3, "(2,3)")
plt.text(4 + offset, 6, "(4,6)")
plt.text(5 + offset * 2, 10, "(5,10)")
plt.text(10 + offset * 2, 20, "(10,20)")
plt.text(15 - offset * 7, 30, "(15,30)")
plt.text(19 - offset * 7, 36, "(19,36)")
plt.show()
