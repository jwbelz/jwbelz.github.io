#!/usr/bin/python3

# Homework 1, Problem 2
# P4730 F26, JB

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

matplotlib.rcParams.update({'font.size': 14})

from numpy import linalg as LA
from numpy import matrix as M
from numpy.linalg import pinv

# create the data, else read it from a file

datafile = "cosines.dat"

if(False):

    print("\n Creating data in "+datafile)

    # data given with assignment
    
    #N = 100
    #t = np.linspace(0.0,np.pi,N) + 0.1*np.random.normal(0.0,1.0,N)
    #x = 20.0*np.cos(1.0*t) + 8.0*np.cos(4.0*t) + np.random.normal(0.0,1.0,N)
    
    # data for grading 

    N = 100
    t = np.linspace(0.0,np.pi,N) + 0.12*np.random.normal(0.0,1.0,N)
    x = -13.0*np.cos(1.0*t) + 12.0*np.cos(4.0*t) + np.random.normal(0.0,1.0,N)

    fyle = open(datafile, "w")
    for i in range(0,N):
        fyle.write("%4d %8.4f %8.4f \n" % (i,t[i],x[i]))
    fyle.close()
    
else:

    print("\n Reading data from "+datafile)
    data = np.loadtxt(datafile)
    t = data[:,1]
    x = data[:,2]

# plot the raw data

plt.figure(figsize=(8,6))
plt.scatter(t,x)
plt.xlabel("$t$")
plt.ylabel("$x(t)$")
plt.axis([-0.5,3.5,-40.0,40.0])
plt.savefig("cosines_dat.png")
plt.show()

# store inputs as column vectors

tcv = t[:,None]
xcv = x[:,None]

# create column vectors for sinusoidal component 

cos1t = np.cos(1.0*tcv)
cos4t = np.cos(4.0*tcv)

# concatenate to form VDM matrix...

vdm = np.concatenate((cos1t, cos4t), axis=1)

# ... and solve

vdmi = pinv(vdm)
a    = np.transpose(vdmi.dot(xcv))[0]
a1   = a[0]
a4   = a[1]

print("coefficient of cos(1x): %7.4f" % a1)
print("coefficient of cos(4x): %7.4f" % a4)
print("")

# Replot data with fit curve

tfit = np.arange(t[0],t[len(t)-1],0.001)
xfit = a1*np.cos(1.0*tfit) + a4*np.cos(4.0*tfit)

plt.figure(figsize=(8,6))
plt.scatter(t,x,label="data")
plt.plot(tfit,xfit,label="fit")
plt.xlabel("$t$")
plt.ylabel("$x(t)$")
plt.axis([-0.5,3.5,-40.0,40.0])
plt.grid()
plt.legend(loc="upper right")
plt.text(1.4,-24.0,"$a_1 =$"+str(a1)[0:5])
plt.text(1.4,-28.0,"$a_4 =$"+str(a4)[0:5])

plt.savefig("cosines.png")
plt.show()

#
