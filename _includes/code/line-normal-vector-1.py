import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines

ax = plt.axes()

ax.arrow(0.0, 0.0, 6.0, 4.0, head_width=0.5, head_length=0.7, fc='lightblue', ec='black')
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
plt.title("A Line and its Normal Vector",fontsize=10)
labelLines(plt.gca().get_lines(),align=True,fontsize=14)
plt.text(6, 3, "(6,4)")
# plt.savefig('how_to_plot_a_vector_in_matplotlib_fig1.png', bbox_inches='tight')
plt.show()
# plt.close()
