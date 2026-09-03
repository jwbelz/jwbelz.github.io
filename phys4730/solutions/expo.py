#!/usr/bin/python3

import numpy as np
import scipy.optimize as SciOpt
from numpy import random as rnd
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import stats

############################
### PHYS 4730/6720 lab04 ###
############################

### generate or read the data ###

datafile = "expodata.dat"

if(False): 

    # python uses pdf = (1/beta)*exp(-x/beta)
    
    beta = 5.0
    N = 10000
    x  = rnd.exponential(beta, size=N)
    
    fyle = open(datafile, "w")
    for i in range(0,len(x)):
        fyle.write("%12.8f \n" % x[i])
    fyle.close()

else:
    
    x = np.loadtxt(datafile)

### find beta from mean and variance = sqrt(standard deviation) ###
    
xmean = np.mean(x)
xstdv = np.std(x)
xskew = stats.skew(x,axis=0,bias=True)
xkurt = stats.kurtosis(x,axis=0,bias=True)

print("")
print("beta from mean = %7.4f" % xmean)
print("beta from std  = %7.4f" % xstdv)
print("")
print("skewness       = %7.4f" % xskew)
print("kurtosis       = %7.4f" % xkurt)
print("")
print("The parameter beta cannot be determined from" )
print("skewness or kurtosis, as these are the same"  )
print("(2 and 6, respectively) for all exponential"  )
print("distributions."                               ) 
print("")

### plot pdf, assuming beta = the mean of the distribution ###

beta = xmean

aa = 0.0
bb = 20.0

n,b,p = plt.hist(x, bins=100, range=[aa,bb],density=True,label="data") 
plt.xlabel('$x$')
plt.ylabel('empirical pdf')
plt.grid()

dist = stats.expon(0.0,beta)
xx = np.arange(aa,bb,0.01)
p  = dist.pdf(xx)
plt.plot(xx,p,color='r',label="pdf")
plt.legend(loc="upper right")

plt.savefig("expo.png")
plt.show()


#
