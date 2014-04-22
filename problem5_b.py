from __future__ import division
import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt

def plot_init(x, y, t):
	plt.clf()
	plt.xlabel(x)
	plt.ylabel(y)
	plt.title(t)
	plt.hold(True)

def plot_draw(X, Y, case):
	plt.plot(X, Y, label="h = %s" % case)
	plt.legend(loc="best")

def plot_loglog(X, Y):
	plt.loglog(X, Y)

def f(t, y):
	return - 200 * t * (y**2)

def backward_euler(t0, y0, h):
	y = y0
	t = t0
	y_arr = [y0]
	t_arr = [t0]
	while(t < 1):
		y = opt.newton(back, y, args = (t + h,y))
		t = t + h
		y_arr.append(y)
		t_arr.append(t)
	return t_arr, y_arr

def back(y, t, oldy):
	return f(t, y) + oldy - y

h_arr = []
error_arr = []
y_t1 = 1 / 101

plot_init("t", "y_h", "Plot for Problem5 b) Backward Euler")

h = 0.015625
t,y = backward_euler(0, 1, h)
plot_draw(t, y, h)
h_arr.append(h)
error_arr.append(np.abs(y[-1] - y_t1))

h = 0.03125
t,y = backward_euler(0, 1, h)
plot_draw(t, y, h)
h_arr.append(h)
error_arr.append(np.abs(y[-1] - y_t1))

h = 0.0625
t,y = backward_euler(0, 1, h)
plot_draw(t, y, h)
h_arr.append(h)
error_arr.append(np.abs(y[-1] - y_t1))

h = 0.125
t,y = backward_euler(0, 1, h)
plot_draw(t, y, h)
h_arr.append(h)
error_arr.append(np.abs(y[-1] - y_t1))

h = 0.25
t,y = backward_euler(0, 1, h)
plot_draw(t, y, h)
h_arr.append(h)
error_arr.append(np.abs(y[-1] - y_t1))

h = 0.5
t,y = backward_euler(0, 1, h)
plot_draw(t, y, h)
h_arr.append(h)
error_arr.append(np.abs(y[-1] - y_t1))

h = 1
t,y = backward_euler(0, 1, h)
plot_draw(t, y, h)
h_arr.append(h)
error_arr.append(np.abs(y[-1] - y_t1))
plt.savefig("problem5_b.png")

plot_init("h", "error", "Plot for Problem5 b) error versus h")
plot_loglog(h_arr, error_arr)
plt.savefig("problem5_b_error.png")