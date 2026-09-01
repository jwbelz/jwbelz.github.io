#!/usr/plocal/bin/python3

import math
import sympy
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
import os 
import sys
from numpy.linalg import pinv

### read and parse the data ###

data = np.loadtxt('quaddata.dat')
pt   = data[:,0]
xx   = data[:,1]
yy   = data[:,2]

### print and plot to make sure it looks OK ###

if(False):

    for i in range(len(pt)):
        print(i,xx[i],yy[i])

    plt.scatter(xx,yy)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis([0.0,12.0,-20.0,10.0])
    plt.show()

### set up the Vandermonde matrix ###

# we need the x and y-axes as column vectors
xcv = xx[:, None]
ycv = yy[:, None]

# now we can create the matrix
vdm = np.concatenate((xcv**0, xcv**1, xcv**2), axis=1)

if(False):
    print(vdm)

### solve to obtain best fit ###

a = np.transpose(pinv(vdm).dot(ycv))[0]

### output numerical results and create plot ###

print("")
for i in range(len(a)):
    print("a[%1d] = %8.4f " % (i,a[i]))
print("")

xxx = np.arange(xx[0],xx[len(xx)-1],0.01)
yyy = a[0] + a[1]*xxx + a[2]*xxx*xxx

plt.scatter(xx,yy,s=20.0,color='k',marker='o',label="data points")
plt.plot(xxx,yyy,color='r',label="fit result")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis([0.0,12.0,-20.0,10.0])
plt.legend()

plt.savefig("findquadratic.png")
plt.show()

#
