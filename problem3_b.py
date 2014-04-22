from __future__ import division
import numpy as np
import numpy.linalg as la
import scipy.special as special
import scipy.integrate as integrate

def gauss_quad(func, n):
    A = np.zeros((n,n))
    b = np.zeros((n,1))
    for i in range(n): 
        b[i][0]=(1**(i+1)-(-1)**(i+1))/(i+1)

    approx = 0
    nodes = special.legendre(n).weights[:,0]
    for i in range(n):
        for j in range(n):
            A[i][j] = nodes[j]**i
    
    w = la.solve(A,b)

    for i in range(n):
        approx = approx + func(nodes[i])*w[i]

    return abs(approx - integrate.quad(lambda x: func(x),-1.0,1.0)[0])