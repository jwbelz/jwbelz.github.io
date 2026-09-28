#!/usr/bin/python3

### PHYS 4730 F2025      ###
### HW02 Problem 1       ###

import numpy as np
import scipy.optimize as SciOpt
from numpy import random
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import sys

matplotlib.rcParams.update({'font.size': 18})

### parse the command line ###

if(len(sys.argv) != 4):
    print("Usage: gengauss.py [mu] [sigma] [nrv]")
    exit(1)

MOO = float(sys.argv[1])
SIG = float(sys.argv[2])
NRV = int(sys.argv[3])

print(" ")
print("Generating Normal RV with mean = %7.2f" % MOO)
print("                         width = %7.2f" % SIG)
print("                             N = %7d"   % NRV)

### function for the PDF ###

def PDF(x):
    return np.exp(-0.5*(x-MOO)**2/SIG**2)/np.sqrt(2.0*np.pi*SIG*SIG)

### use central limit theorem to generate a gaussian... ###

M = 10            # number of uniform variates to average
q = np.zeros(M)   # holder for variates we're averaging
x = np.zeros(NRV) # the Gaussian RV   

for i in np.arange(0,NRV):
    q = random.uniform(0.0,1.0,M)
    x[i] = (np.mean(q)-0.5)*np.sqrt(12.0*M)*SIG + MOO

### ...and the pdf ###

xmin = MOO - 5.0*SIG
xmax = MOO + 5.0*SIG 
xx   = np.arange(xmin,xmax,0.001)
pdf  = PDF(xx)

### make the plot ### 
    
plt.figure(figsize=(8,8))

plt.hist(x,100,density=True,color='r',label="Normal RV")
plt.plot(xx,pdf,color='k',linewidth=3,label="$f(x|\mu,\sigma)$")

plt.xlabel("$x$")
plt.xlim(xmin,xmax)
plt.ylabel("(E)PDF")
plt.legend(loc="upper right")
titletext = "$\\mu$ = %5.2f, $\\sigma$ = %5.2f, N = %6d" % (MOO,SIG,NRV)
plt.title(titletext)
plt.savefig("gengauss.png")
plt.show()
    
print(" ")
        
#
