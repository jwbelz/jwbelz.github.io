#!/usr/plocal/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special  import gammaincc

######################
def GR(x,a0,a1,a2,a3):
######################

    return a0 + a1*x + a2*x*x + a3*x*x*x

###########################
def GRpp(x,a0,a1,a2,a3,a4):
###########################

    return a0 + a1*x + a2*x*x + a3*x*x*x + a4*x*x*x*x

###################
### main script ###
###################

### read the data ###

npeak = np.arange(1,8,1)
tpeak = np.array([0.31172,
                  0.33927,
                  0.36113,
                  0.38370,
                  0.40201,
                  0.41481,
                  0.42316])
speak = np.array([0.0015,
                  0.0015,
                  0.0005,
                  0.0005,
                  0.0005,
                  0.0005,
                  0.0005])
 
### first, fit the curve ###

popt, pcov = curve_fit(GR,  npeak,tpeak,sigma=speak)
qopt, qcov = curve_fit(GRpp,npeak,tpeak,sigma=speak)

#print(popt)
#print(pcov)

### Compute chi square ###

gr_a0 = popt[0]
gr_a1 = popt[1]
gr_a2 = popt[2]
gr_a3 = popt[3]

gr_sa0 = np.sqrt(pcov[0,0])
gr_sa1 = np.sqrt(pcov[1,1])
gr_sa2 = np.sqrt(pcov[2,2])
gr_sa3 = np.sqrt(pcov[3,3])

GR_t_expected = GR(npeak,gr_a0,gr_a1,gr_a2,gr_a3)
GR_chisq = np.sum(((tpeak-GR_t_expected)/speak)**2)
GR_df = len(npeak) - 4

grpp_a0 = qopt[0]
grpp_a1 = qopt[1]
grpp_a2 = qopt[2]
grpp_a3 = qopt[3]
grpp_a4 = qopt[4]

grpp_sa0 = np.sqrt(qcov[0,0])
grpp_sa1 = np.sqrt(qcov[1,1])
grpp_sa2 = np.sqrt(qcov[2,2])
grpp_sa3 = np.sqrt(qcov[3,3])
grpp_sa4 = np.sqrt(qcov[4,4])

GRpp_t_expected = GRpp(npeak,grpp_a0,grpp_a1,grpp_a2,grpp_a3,grpp_a4)
GRpp_chisq = np.sum(((tpeak-GRpp_t_expected)/speak)**2)
GRpp_df = len(npeak) - 5

### Compute quality factor ###

GR_quality   = gammaincc(GR_df/2.0,  GR_chisq/2.0  )
GRpp_quality = gammaincc(GRpp_df/2.0,GRpp_chisq/2.0)

### Output results in the form requested ###

print(" ")
print("best-fit value of a0 from GR:   %7.5f +- %7.5f" % (gr_a0,gr_sa0))
print("best-fit value of a0 from GR++: %7.5f +- %7.5f" % (grpp_a0,grpp_sa0))
print("chi-square from GR:             %7.5f" % (GR_chisq))
print("chi-square from GR++:           %7.5f" % (GRpp_chisq))
print("Goodness=of=fit q from GR:      %7.5f" % (GR_quality))
print("Goodness=of=fit q from GR++:    %7.5f" % (GRpp_quality))
if(GR_quality > GRpp_quality):
    print("best-pick model: GR")
else:
    print("best-pick model: GR++")
print(" ")

print(" ")

### as a check, plot the data with error bars along with the fit results ###

if(True):

    fig = plt.figure(figsize=(6,8))

    fig.add_subplot(211)
    
    plt.errorbar(npeak,tpeak,yerr=speak,fmt = 'o', label="data")
    n_smooth  = np.arange(0.0,1.1*npeak[len(npeak)-1],0.1)
    gr_smooth = GR(n_smooth,gr_a0,gr_a1,gr_a2,gr_a3)
    plt.plot(n_smooth,gr_smooth,label='GR fit')
    plt.xlabel("peak")
    plt.ylabel("time")
    plt.axis([0.0,8.0,0.2,0.5])
    plt.legend(loc="upper left")

    fig.add_subplot(212)

    plt.errorbar(npeak,tpeak,yerr=speak,fmt = 'o', label="data")
    n_smooth    = np.arange(0.0,1.1*npeak[len(npeak)-1],0.1)
    grpp_smooth = GRpp(n_smooth,grpp_a0,grpp_a1,grpp_a2,grpp_a3,grpp_a4)
    plt.plot(n_smooth,grpp_smooth,label='GR++ fit')
    plt.xlabel("peak")
    plt.ylabel("time")
    plt.axis([0.0,8.0,0.2,0.5])
    plt.legend(loc="upper left")

    plt.show()

#
