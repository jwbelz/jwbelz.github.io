#!/usr/plocal/bin/python3

# Quantile calculator
# 
# PHYS4730, Fall 2026
# 
# HW04 Problem 1
# 
# JB's solution.

import numpy as np
import scipy.optimize as SciOpt
import scipy.stats as SciStat
import scipy.special as SciSpec
from numpy import random as rnd
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
import matplotlib
import sys

matplotlib.rcParams.update({'font.size': 12})

### flag for making diagnostic plots ###

LDEBUG = True

### take the quantile of interest from the command line ###

if(len(sys.argv) != 2):
    print("you need to specify a quantile")
    quit()
    
alpha = float(sys.argv[1])

### read the data file ###

datafile = "rvvec.dat"
rvvec = np.loadtxt(datafile)

### plot the RV distribution ###

if(LDEBUG):
    print("Finding %5.3f quantile" % alpha)
    fig = plt.figure()
    plt.hist(rvvec,bins=30,range=[0.0,3.0])
    plt.xlabel("random variable $x$")
    plt.show()

### find empirical quantile ###

quant_data = np.quantile(rvvec,alpha)

### define function to compute empirical likelihood ratio ### 

def ELR(data,valu,alph): 
    
    # Define estimating function.
    
    g = (data < valu) - alph 
    
    # Set up the Lagrange multiplier condition for L
    
    constraintEQN = lambda L : sum(np.divide(g,1 + g*L))
    
    # Find the multiplier
    
    Lam = SciOpt.fsolve(constraintEQN,0);
    
    # Return empirical likelihood ratio
    
    return 2*np.sum(np.log(1+Lam*g))

### back to main script ### 

### Pick some reasonable range for possible values of the quantile ###

pmrange = 0.04
x = np.linspace(quant_data-pmrange, quant_data+pmrange,100)

### create y to holdELR values ###

y = np.zeros(len(x))

### compute ELR values... ###

for ii in np.arange(0,len(x)):
    y[ii] = ELR(rvvec,x[ii],alpha) 
    
### ...and make note of minimum ELR, 95% C.I. line ###
    
cen = x[np.argmin(y)]
y95 = np.ones(len(x))*chi2.ppf(0.975, df=1)

### make a plot ###

if(LDEBUG):
    fig = plt.figure(figsize=(8,6))

    plt.plot(x,y,label='EL Ratio', linewidth=3)
    plt.plot(x,y95,'r',label='95% C.I. cutoff',linewidth=3)
    plt.xlabel('Possible values of %5.3f quantile' % alpha)
    plt.ylabel('EL ratio')
    plt.legend()
    plt.show()

### find intersections of the curves ###

# Define cost function which sets target ELR based on confidence level

ELRiterator = lambda p : ELR(rvvec,p,alpha) - chi2.ppf(0.975, df=1) 

# Use brent method to find intersection point for ELR and cutoff

limHi = SciOpt.brentq(ELRiterator, cen,         cen+pmrange)   # RHS
limLo = SciOpt.brentq(ELRiterator, cen-pmrange, cen)           # LHS

### final output for user ###

print(" ")
print("The %5.3f quantile of the data is: %8.6f" % (alpha,quant_data))
print("The 95 pct CI is:                 (%8.6f, %8.6f)" % (limLo,limHi))
print(" ")

#
