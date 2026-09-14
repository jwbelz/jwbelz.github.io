#!/usr/bin/python3

# Homework 1, Problem 3
# P4730 F26, JB

import numpy as np
from numpy import linalg as LAn
from scipy import linalg as LAs
from numpy import matrix as M
from numpy import random as rnd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 14})

### create (or read) the data ###

datafile = "polyfile.dat"

if(False):
    
    print("")
    print("Creating data in "+datafile)

    N = 51

    # Gaussian parameters

    # data one

    #a0 =  4.3
    #a1 =  3.0
    #a2 = -1.5
    #noise = 1.2

    # data two
    
    a0 =  -3.5
    a1 =   2.0
    a2 =   1.7
    noise = 1.1

    # domain and range

    x       = np.linspace(-2.0,2.0,N)
    fnaught = a0 + a1*x + a2*x*x
    f       = fnaught + rnd.normal(0,noise,N);
    
    fyle = open("polyfile.dat", "w")
    for i in range(0,len(x)):
        fyle.write("%4d %8.4f %8.4f \n" % (i,x[i],f[i]))
    fyle.close()

else:
    
    print("")
    print("Reading data from "+datafile)
    print("")
    data = np.loadtxt(datafile)
    x = data[:,1]
    f = data[:,2]

### plot the raw data ###

plt.figure(figsize=(9,7))
plt.scatter(x,f)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.axis([-2.5,2.5,-10.0,10.0])

plt.savefig("polyfile.png")
plt.show()
    
### set up the Vandermonde matrix, 2nd order in x ###

### convert x and g to column vectors ###

xcv = x[:,None]
fcv = f[:,None]

### now build the matrix ###

vdm = np.concatenate((xcv**0, xcv**1, xcv**2), axis=1)

### invert the Vandermonde matrix and solve ###

vdmi = np.linalg.pinv(vdm)
a = np.dot(vdmi,fcv);

a0_est = a[0][0]
a1_est = a[1][0]
a2_est = a[2][0]

print("polynomial coefficients:")
print("")
for i in np.arange(0,len(a)):
    print("%2i %16.12f" % (i,a[i][0]))

### plot the results ###

plt.figure(figsize=(9,7))
plt.scatter(x,f,label="data")
plt.plot(x,a[0]+a[1]*x+a[2]*x*x,label="fit result")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.legend(loc="lower right")
plt.axis([-2.5,2.5,-10.0,10.0])

plt.savefig("polyfit.png")
plt.show()

### One more thing.. use bootstrap method to ###
### guesstimate uncertainties in parameters. ###

noise = np.std(a[0]+a[1]*x+a[2]*x*x - f)
print("")
print("Estimated noise:          %8.4f" % noise)

N = len(x)
Ntrials = 1000

a0_arr = np.zeros(Ntrials)
a1_arr = np.zeros(Ntrials)
a2_arr = np.zeros(Ntrials)

fnaught = a[0] + a[1]*x + a[2]*x*x

for i in range(0,Ntrials):
    
    f = fnaught + rnd.normal(0,noise,N);
    fcv = f[:,None]
    a = np.dot(vdmi,fcv);

    a0_arr[i] = a[0][0]     
    a1_arr[i] = a[1][0]     
    a2_arr[i] = a[2][0]

print("")
print("coefficients with uncertainties:")
print("")
print("a0:  %7.4f +- %6.4f" % (a0_est,np.std(a0_arr)))
print("a1:  %7.4f +- %6.4f" % (a1_est,np.std(a1_arr)))
print("a2:  %7.4f +- %6.4f" % (a2_est,np.std(a2_arr)))
print("")

#
