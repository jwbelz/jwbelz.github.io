#!/usr/bin/python3

# Prepare a standalone Python script gaussplot.py that
#
# * Generates 10,000 random numbers in a Gaussian distribution with a
#   mean <x> = 10.0 and a width sigma_x = 3.0
#
# * Sorts these numbers into bins of width Delta-x = 0.2. A convenient
#   way of doing this is with the numpy.histogram function.
#
# * Plots the contents of each bin, with sqrt(N) errorbars, versus the
#   center of the bin. Output this plot to the file gaussplot.png

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 16})

### generate the random numbers ###

mu     = 10.0
sigmax = 3.0
N      = 10000

data = np.random.normal(mu,sigmax,10000)

### sort random numbers into bins ###

xmin   =  0.0
xmax   = 20.0
deltax = 0.2

nbins = int((xmax-xmin)/deltax)

n,b = np.histogram(data,bins=nbins,range=[xmin,xmax])
nerr = np.sqrt(n)

bcenter = np.zeros(len(n))

for i in np.arange(0,len(n)):
    bcenter[i] = 0.5*(b[i+1]+b[i])
    if(False): print("%5.1f %5.1f" %(bcenter[i],n[i]))
    
### plot contents of each bin with sqrt(N) errorbars, output to file ###

plt.figure(figsize=(8, 8))

plt.errorbar(bcenter,n,yerr=nerr,fmt='o',markersize=5.0,ls='none',linewidth=2, capsize=6,c='k',label="binned data")
plt.xlabel("x")
plt.ylabel("events per ($\\Delta x =$ %3.2f)" % deltax)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("gaussplot.png",transparent=False, facecolor='white')
plt.show()


#
