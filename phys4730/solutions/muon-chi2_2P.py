#!/usr/bin/python3

### PHYS 4730 F2026   ###
### HW02 Problem 3    ###

### TWO PARAMETER FIT VERSION ###

import numpy as np
import matplotlib.pyplot as plt
from scipy.special  import gammaincc
from scipy.optimize import curve_fit
import matplotlib

matplotlib.rcParams.update({'font.size': 14})

#########################
def chi2(x,y,sigy,c,tau):
#########################

    x2 = 0.0
    for i in range(0,len(x)):
        x2 = x2 + pow((y[i] - c*np.exp(-1.0*x[i]/tau))/sigy[i],2.0)
    return x2

################
def func(x,a,b):
################
    
    return a*np.exp(-x/b)

###################
### main script ###
###################

### open and read the binned data file ###

filename = "muons20051130.dat"
t = np.loadtxt(filename,usecols=(0))

### use matplotlib histogram to sort times into bins ###

nbins =   25
tmin  =  0.0
tmax  =  8.0
dt    = (tmax-tmin)/float(nbins)

n, b, p = plt.hist(t,nbins,range=(tmin,tmax))
plt.show()

### assign x, y, sigy from the binned data ###

b = np.delete(b,nbins)

x    = b + 0.5*dt
y    = n
sigy = np.sqrt(n)

plt.errorbar(x,y,sigy,fmt='o')
plt.show()

### use scipy.optimize.curve_fit to extract tau ###

popt, pcov = curve_fit(func, x, y, sigma=sigy, method='lm')

aaa_fit = popt[0]
tau_fit = popt[1]
tau_err = np.sqrt(pcov[1,1])

### compute chi**2 and goodness-of-fit ###

chisq = np.sum((y-func(x,aaa_fit,tau_fit))**2/sigy**2)
ndf   = len(x) - len(popt)
Q     = gammaincc(0.5*ndf,0.5*chisq)

### output results ###

print("")
print("Best Muon Lifetime: %6.4f +- %6.4f microseconds" % (tau_fit,tau_err)) 
print("chisquare/df:       %6.3f/%2d" % (chisq,ndf))
print("goodness-of-fit:    %8.6f" % Q)
print("")

### make the final figure ### 

plt.figure(figsize=(8,8))

tt = np.arange(tmin,tmax,0.001)
ff = func(tt,aaa_fit,tau_fit)

plt.errorbar(x,y,sigy,fmt='o',label='data')
plt.xlabel("time ($\\mu sec$)")
plt.ylabel("muons/bin")
labeltext = "best fit: $\\tau$ = %5.3f $\pm$ %5.3f" % (tau_fit,tau_err)
plt.plot(tt,ff,label=labeltext)
plt.xlim(tmin,tmax)
plt.ylim(0.0,1.1*np.max(y))
plt.grid()
plt.legend(loc="upper right")

plt.savefig("muon-chi2_2P.png")
plt.show()

#
