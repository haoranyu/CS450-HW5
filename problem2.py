from __future__ import division
import numpy as np
import scipy as sp
import scipy.integrate as inte
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import math

def plot_init(x, y, t):
	plt.clf()
	plt.xlabel(x)
	plt.ylabel(y)
	plt.title(t)
	plt.hold(True)

def plot_draw(X, Y, case):
	plt.plot(X, Y, label="%s" % case)
	plt.legend(loc="best")

def plot_loglog(X, Y, case):
	plt.loglog(X, Y, label="%s" % case)
	plt.legend(loc="best")

def midpoint(F, x):
	m = 0
	for i in range(0, len(x)-1):
		m += fixed_quad(F, x[i], x[i+1], n=1)
	return m

def fixed_quad(f, a, b, n):
    x,w = sp.special.j_roots(n, 0.0, 0.0)
    x = np.real(x)
    y = (b-a)*(x+1)/2.0 + a
    ret = (b-a)/2.0*sum(w*f(y),0)
    return ret

def trapezoid(F, x):
    y = F(x)
    y = np.array(np.copy(y))
    x = np.array(np.copy(x))
    d = np.diff(x)
    n = len(y.shape)
    slice1 = [slice(None)] * n
    slice2 = [slice(None)] * n
    slice1[-1] = slice(1, None)
    slice2[-1] = slice(None, -1)
    ret = (d * (y[slice1] + y[slice2]) / 2.0).sum(-1)
    return ret

def simpson(F, x):
    y = F(x)
    y = np.array(np.copy(y))
    x = np.array(np.copy(x))
    n = len(y.shape)
    N = y.shape[-1]

    all = (slice(None),)*n
    sl = []
    for i in range(3):
    	l = list(all)
    	l[-1] = slice(i, N-(2-i), 2)
    	sl.append(tuple(l))

    h = np.diff(x)
    h0 = h[sl[0]]
    h1 = h[sl[1]]
    hadd = h0 + h1
    htime = h0 * h1
    hdiv = h0 / h1
    ret = np.add.reduce(hadd/6.0*(y[sl[0]]*(2-1.0/hdiv) + y[sl[1]]*hadd*hadd/htime + y[sl[2]]*(2-hdiv)),-1)
    return ret

def monte_carlo(F, x, n):
    sum = 0.0
    x = np.random.rand(n)
    for item in x:
        sum = sum + f(item)
    I = sum/n
    return I

def f(x):
    return 4.0 / (1+x**2)

def evaluate(F, X, n):
	t = trapezoid(F, X)
	s = simpson(F, X)
	m = midpoint(F, X)
	c = monte_carlo(F, X, n)
	return m,t,s,c

def empirical_order(H, E):
	n = len(H)
	p_arr = []
	for i in range(n-1):
		H1 = H[i]
		H2 = H[i+1]
		E1 = E[i]
		E2 = E[i+1]
		p = np.log(E2/E1) / np.log(H2/H1)
		p_arr.append(p)
	return np.sum(p_arr)/n


np.random.seed(3)

k_s = np.arange(2,1000,10)
h_arr = np.zeros(k_s.size)
mid = np.zeros(k_s.size)
trap = np.zeros(k_s.size)
simp = np.zeros(k_s.size)
monc = np.zeros(k_s.size)

for i,npts in enumerate(k_s):
	h = 1.0/(npts+1)
	h_arr[i] = h
	X = np.linspace(0.,1.,npts+1)

	m,t,s,c = evaluate(f, X, (npts+1))
	mid[i] = m; trap[i] = t; simp[i] = s; monc[i] = c

plot_init("h", "Solution of pi", "Approximations for pi various h values")
plot_draw(h_arr, mid, "Midpoint rule")
plot_draw(h_arr, trap, "Trapezoid rule")
plot_draw(h_arr, simp, "Simpson rule")
plot_draw(h_arr, monc, "Monte Carlo")
plt.savefig("problem2_approx.png")

plot_init("h", "Error", "Error in Quadrature Rules")
plot_loglog(h_arr, abs(mid -np.pi), "Midpoint rule")
plot_loglog(h_arr, abs(trap-np.pi), "Trapezoid rule")
plot_loglog(h_arr, abs(simp-np.pi), "Simpson rule")
plot_loglog(h_arr, abs(monc-np.pi), "Monte Carlo")
plt.savefig("problem2_error.png")

print "EOC for Midpoint rule: %g " % empirical_order(h_arr, mid)
print "EOC for Trapezoid rule: %g " % empirical_order(h_arr, trap)
print "EOC for Simpson rule: %g " % empirical_order(h_arr, simp)
print "EOC for Monte Carlo: %g " % empirical_order(h_arr, monc)