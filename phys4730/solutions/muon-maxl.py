#!/usr/bin/python3

### PHYS 4730 F2025   ###
### HW02 Problem 4    ###

import numpy as np
import matplotlib.pyplot as plt
from scipy.special  import gammaincc
from scipy.optimize import curve_fit
from scipy import optimize
import matplotlib

matplotlib.rcParams.update({'font.size': 14})

#########################
def llikely(t,tau,t1,t2):
#########################

    A = 1.0/((np.exp(-t1/tau)-np.exp(-t2/tau))*tau)  # normalize PDF to 1.0
 
    llf  = 0.0    
    for i in range(0,len(t)):
        llf = llf + (np.log(A) - t[i]/tau) 
        
    return llf

###################
### main script ###
###################

### read data and set up ###

filename = "muons20051130.dat"
t_data   = np.loadtxt(filename)

t1 = 0.0
t2 = 8.0

taumin   =  1.70    
taumax   =  2.30
ntaustep =  600
taustep  = (taumax-taumin)/float(ntaustep)

mmax      = -999999.0
tau_atmax = -999999.0
a_atmax   = -999999.0

tau = np.arange(taumin,taumax,taustep)
mmm = np.zeros(ntaustep)

### loop through values of tau, ###
### find maximum log(L)         ###

for j in range(0,ntaustep):

    mmm[j] = llikely(t_data,tau[j],t1,t2)

    if(mmm[j] > mmax):
        mmax      = mmm[j]
        tau_atmax = tau[j]

### Now, find uncertainty in tau estimate.       ###
### Find values for which log(L) changes by 1/2. ###

def llikely_err(tau):    
    return llikely(t_data,tau_atmax,t1,t2) - llikely(t_data,tau,t1,t2) - 0.5

### Find +-1 sigma tau with and without cutoff ###

taup1  = optimize.brentq(llikely_err, tau_atmax, taumax)
taum1  = optimize.brentq(llikely_err, taumin,    tau_atmax)

print("")
print("Best tau: (%6.4f + %6.4f - %6.4f) microseconds" % (tau_atmax, taup1-tau_atmax,  tau_atmax-taum1  ) )

### diagnostic plot of log-likelihood vs tau ###

plt.figure(figsize=(9,5))

plt.plot(tau,mmm)
plt.plot([taumin,taumax],[mmax,mmax])         # line thru max 
plt.plot([taumin,taumax],[mmax-0.5,mmax-0.5]) # line thru max-0.5
plt.plot([taum1,taum1],[-10000.0,0.0])
plt.plot([taup1,taup1],[-10000.0,0.0])
plt.axis([1.9,2.1,-6764.0,-6755.5])

plt.xlabel("$\\tau$ (microseconds)")
plt.ylabel("log-likelihood")
plt.grid()
plt.show()

### results to standard output ###

print(" ")
print("maximum likelihood fit results: ")
print(" ")
print("    tau = %1.3f + %1.3f - %1.3f" % (tau_atmax,(taup1-tau_atmax),(tau_atmax-taum1)) )
print("   MMAX = %5.5f " % mmax)
print(" ")

### final plot of result with data ###

binwidth = 0.32

nentries, edges, patches = plt.hist(x=t_data, bins=np.arange(t1,t2+binwidth, binwidth), color='#0504aa', alpha=0.7, rwidth=0.85)
plt.show()

### prepare plot of points with errorbars ###

nerror = np.sqrt(nentries)
bintime = np.zeros(len(nentries))

for i in range(0,len(nentries)):
    bintime[i] = edges[i] + 0.5*binwidth

### normalize nentries and uncertainty to unit area ###
### scale fit curve to account for binwidth         ###
    
a_atmax = 1.0/((np.exp(-t1/tau_atmax)-np.exp(-t2/tau_atmax))*tau_atmax)
  
tot_nentries = np.sum(nentries)
nentries     = nentries/tot_nentries
nerror       = nerror/tot_nentries
binfunc      = binwidth*a_atmax*np.exp(-bintime/tau_atmax)

### make the final plot ###

plt.figure(figsize=(8,8))
    
plt.errorbar(bintime,nentries,nerror,fmt='o',label="binned data")
labeltext = "best fit: $\\tau$ = %5.3f + %5.3f - %5.3f" % (tau_atmax,taup1-tau_atmax,tau_atmax-taum1)
plt.plot(bintime,binfunc,label=labeltext)
plt.xlabel("time (microseconds)")
plt.ylabel("normalized counts per bin")
plt.grid()
plt.legend()

plt.savefig("muon-maxl.png")
plt.show()

#
