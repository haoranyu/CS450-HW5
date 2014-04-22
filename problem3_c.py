from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from problem3_b import gauss_quad

def f(x):
    return np.sin(2.0*np.pi*x)

def g(x):
    return abs(x)

def plot_init(x, y, t):
	plt.clf()
	plt.xlabel(x)
	plt.ylabel(y)
	plt.title(t)
	plt.hold(True)

def plot_draw(X, Y, case):
	plt.semilogy(X, Y, '-')

err1 = []
err2 = []   
x = np.linspace(1,100,100)

for n in range(1,101):
    err1.append(gauss_quad(f,n))
    err2.append(gauss_quad(g,n))

plot_init("Order", "Error", "Problem 3 c): Error in the Gaussian quadrature versus the order")
plt.semilogy(x, err1)
plt.semilogy(x, err2)
plt.legend(['f(x) = sin(2*pi*x)','g(x) = |x|'],loc = 'best')
plt.savefig("problem3_c.png")