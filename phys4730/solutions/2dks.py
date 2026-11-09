#!/usr/plocal/bin/python3

# 2D KS Test: Separation of Gaussian RVs
# 
# PHYS4730, Fall 2026
# 
# HW04 Problem 2
# 
# Reference: Press and Teukolsky, "Kolmogorov-Smirnov Test for
# Two-Dimensional Data", Computers in Physics <b>2</b>, 74 (1988). 
# 
# JB's solution.

import numpy as np
import scipy.optimize as SciOpt
import scipy.stats as SciStat
from numpy import random as rnd
from numpy import linalg as LA
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'font.size': 12})

######################
def quadct(x,y,xx,yy):
######################
    
### function to return normalized ###
### fraction in each quadrant     ###

    nn = len(xx)
    
    na,nb,nc,nd = 0.0,0.0,0.0,0.0
    
    for k in np.arange(0,nn):
        if(yy[k] > y):
            if(xx[k] > x):
                na += 1.0
            else:
                nb += 1.0
        else:
            if(xx[k] > x):
                nd += 1.0
            else:
                nc += 1.0
        
    fa = na/nn
    fb = nb/nn
    fc = nc/nn
    fd = nd/nn
    
    return fa,fb,fc,fd

#############
def QKS(lam):
#############
    
### function to generate KS probability ###
### (Eq 3 of Press and Teukolsky)       ###

    nterms = 20
    temp   = 0.0
    
    for j in range(1,nterms):
        temp += pow(-1.0,j-1)*np.exp(-2.0*j*j*lam*lam)
    
    return 2.0*temp

########################
def ks2d2s(x1,y1,x2,y2):
########################
    
### function to generate 2D KS statistic ###

    n1 = len(x1)
    n2 = len(x2)
    
    # use points in first sample as origins 
    
    d1 = 0.0
    j1 = -99
    
    for j in np.arange(0,n1):
        
        fa,fb,fc,fd = quadct(x1[j],y1[j],x1,y1)
        ga,gb,gc,gd = quadct(x1[j],y1[j],x2,y2)
        
        test = np.max([np.abs(fa-ga),np.abs(fb-gb),np.abs(fc-gc),np.abs(fd-gd)])
        if(d1 < test):
            d1 = test
            j1 = j
    
    # use points in second sample as origins
    
    d2 = 0.0
    j2 = -99
    
    for j in np.arange(0,n2):
        
        fa,fb,fc,fd = quadct(x2[j],y2[j],x1,y1)
        ga,gb,gc,gd = quadct(x2[j],y2[j],x2,y2)
        
        test = np.max([np.abs(fa-ga),np.abs(fb-gb),np.abs(fc-gc),np.abs(fd-gd)])
        if(d2 < test):
            d2 = test
            j2 = j
    
    # get the average KS statistic
    
    d = 0.5*(d1+d2)
    
    # a factor we'll need to get probability
    
    sqen = np.sqrt(n1*n2/(n1+n2))          # Eq 9 P&T
    
    # get linear correlation coefficient for each sample
    
    r1 = np.corrcoef(x1, y1)[0,1]   
    r2 = np.corrcoef(x2, y2)[0,1]
    rr = np.sqrt(1.0-0.5*(r1*r1+r2*r2))
    
    # get probability from KS probability function
    
    prob = QKS(d*sqen/(1.0+rr*(0.25-0.75/sqen)))            # Eq 8 P&T
    
    # return values
        
    return d, prob

###################
### main script ### 
###################

### Generate our own 2D RVs, to find width    ### 
### difference at which we can 95% confidence ###
### of detection...                           ###

### data set 1 ###

N1   = 100
MU1X = 0.0
MU1Y = 0.0
SG1X = 1.0
SG1Y = 1.0

### data set 2 ###

N2   = 100
MU2X = 0.0
MU2Y = 0.0
SG2X = np.arange(0.4,1.8,0.1)
SG2Y = np.arange(0.4,1.8,0.1)

NAVG   = 100
KSPROB = np.zeros(NAVG)
ksp    = np.zeros(len(SG2X))

print("")

for i in np.arange(0,len(SG2X)):       # loop over sigma_2
     
    for j in np.arange(0,NAVG):        # loop over datasets for averaging
        
        RV1X = rnd.normal(MU1X,   SG1X,N1)
        RV1Y = rnd.normal(MU1Y,   SG1Y,N1)
        RV2X = rnd.normal(MU2X,   SG2X[i],N2)
        RV2Y = rnd.normal(MU2Y,   SG2Y[i],N2)
    
        D,KSPROB[j] = ks2d2s(RV1X,RV1Y,RV2X,RV2Y)
                
    ksp[i] = np.mean(KSPROB)
    print("width of RV2: %5.2f  Mean p-value: %6.4f" % (SG2X[i],ksp[i]))

print("")
    
### ... and make a final plot ###

plt.figure(figsize=(8,8))
plt.subplot(1,1,1)

plt.scatter(SG2X,ksp,label="average p-value")
plt.xlabel("$\\sigma_2$")
plt.ylabel("average p-value")
plt.yscale("log")
plt.plot([0.0,2.0],[0.05,0.05],color='r',linewidth=3,label="5 pct p-value")
plt.axis([0.0,2.0,0.001,1.0])
plt.grid()
plt.legend(loc="upper right")

plt.savefig("2dks.png")
plt.show()

#
