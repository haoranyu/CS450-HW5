from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def plot_init(x, y, t):
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(t)

def system(x, y):
  A = np.array([[1,x[0],x[0]**2,x[0]**3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,x[1],x[1]**2,x[1]**3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,1,x[1],x[1]**2,x[1]**3,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,1,x[2],x[2]**2,x[2]**3,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,x[2],x[2]**2,x[2]**3,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,x[3],x[3]**2,x[3]**3,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,1,x[3],x[3]**2,x[3]**3,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,1,x[4],x[4]**2,x[4]**3,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,x[4],x[4]**2,x[4]**3],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,x[5],x[5]**2,x[5]**3],
              [0,1,2.0*x[1],3.0*x[1]**2,0,-1,-2.0*x[1],-3.0*x[1]**2,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,1,2.0*x[2],3.0*x[2]**2,0,-1,-2.0*x[2],-3.0*x[2]**2,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,2.0*x[3],3.0*x[3]**2,0,-1,-2.0*x[3],-3.0*x[3]**2,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,1,2.0*x[4],3.0*x[4]**2,0,-1,-2.0*x[4],-3.0*x[4]**2],
              [0,0,2,6.0*x[1],0,0,-2,-6.0*x[1],0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,2,6.0*x[2],0,0,-2,-6.0*x[2],0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,2,6.0*x[3],0,0,-2,-6.0*x[3],0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,6.0*x[4],0,0,-2,-6.0*x[4]],
              [0,0,2,6.0*x[0],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,6.0*x[5]]])

  b = np.array([y[0],y[1],y[1],y[2],y[2],y[3],y[3],y[4],y[4],y[5],0,0,0,0,0,0,0,0,0,0])
  return A, b

def f1(coeff,t):
    return coeff[0]+coeff[1]*t+coeff[2]*t**2+coeff[3]*t**3

def f2(coeff,t):
    return coeff[4]+coeff[5]*t+coeff[6]*t**2+coeff[7]*t**3

def f3(coeff,t):
    return coeff[8]+coeff[9]*t+coeff[10]*t**2+coeff[11]*t**3

def f4(coeff,t):
    return coeff[12]+coeff[13]*t+coeff[14]*t**2+coeff[15]*t**3

def f5(coeff,t):
    return coeff[16]+coeff[17]*t+coeff[18]*t**2+coeff[19]*t**3

np.random.seed(3)

x = []
y = []
for i in range(6):
    x.append(np.random.random())
    y.append(np.random.random())

x.sort()
A, b = system(x, y)
coeff = la.solve(A, b)

it = []

for i in range(5):
  it.append(np.linspace(x[i], x[i+1], 20))

y1 = []
y2 = []
y3 = []
y4 = []
y5 = []

for i in range(20):
    y1.append(f1(coeff, it[0][i]))
    y2.append(f2(coeff, it[1][i]))
    y3.append(f3(coeff, it[2][i]))
    y4.append(f4(coeff, it[3][i]))
    y5.append(f5(coeff, it[4][i]))
    
plot_init("x", "y", "Problem1 c)")
plt.plot(x, y, 'kx')

plt.plot(it[0], y1, '-')
plt.plot(it[1], y2, '-')
plt.plot(it[2], y3, '-')
plt.plot(it[3], y4, '-')
plt.plot(it[4], y5, '-')

plt.legend(['Data Points', 'Natural Cubic Spline'])
plt.show()
