from __future__ import division
import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from problem4_b import df

k = range(3,21)
h = np.zeros(18) - k
for i in range(18):
	h[i] = 2**h[i]

def plot_init():
	plt.clf()
	plt.xlabel("h")
	plt.ylabel("error")
	plt.title("Plot for Problem4 c)")
	plt.hold(True)

def plot_draw(X, Y):
	plt.plot(X, Y)

def dfr(x):
	return np.cos(x)

def func(x):
	return np.sin(x)

def measure(f, h):
	x = -1 + h
	x_arr = []
	error_arr = []
	while x < 1:
		x_arr.append(x)
		error = abs(df(f, x, h) - dfr(x))
		error_arr.append(error)
		x += h
	return x_arr, np.max(error_arr)

error = []
for i in range(18):
	x_i, error_i = measure(func, h[i])
	error.append(error_i)

plot_init()
plot_draw(h, error)

plt.savefig("problem4_c.png")