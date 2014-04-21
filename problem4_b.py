from __future__ import division
import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt

def df(f, x, h):
	return (-3 * f(x) + 4 * f(x+h) - f(x+2 * h)) / 2  * h