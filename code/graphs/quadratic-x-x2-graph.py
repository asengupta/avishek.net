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
plt.ylim(-5,5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("$f(x)=x^2$",fontsize=10)
labelLines(plt.gca().get_lines(),align=True,fontsize=14)
plt.xlabel('x')
plt.ylabel('g(x)')

for c in range(0,5):
    fc=y-c
    plt.plot(x, fc, '-r')
    plt.text(0+0.1, -c-0.1, f'G({c})')

plt.show()
# plt.close()
