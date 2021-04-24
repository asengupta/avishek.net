import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines

ax = plt.axes()

x = np.linspace(-5,5,50)
y = x*x
plt.grid()

plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlim(-5,5)
plt.ylim(-1,8)
plt.xlabel('x')
plt.ylabel('f(x)')

plt.gca().set_aspect('equal', adjustable='box')
plt.plot(x, y, '-r')
plt.title("$f(x)=x^2$",fontsize=10)
labelLines(plt.gca().get_lines(),align=True,fontsize=14)
# plt.savefig('how_to_plot_a_vector_in_matplotlib_fig1.png', bbox_inches='tight')
plt.show()
# plt.close()
