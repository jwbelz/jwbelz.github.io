#!/usr/bin/python3

# Homework 1, Problem 1
# P4730 F26, JB

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA
from numpy import matrix as M

import scipy
from scipy.linalg import solve_triangular

matplotlib.rcParams.update({'font.size': 14})

### loop over matrix sizes ###

imax = 10

svtime = np.zeros(imax)
qrtime = np.zeros(imax)
indexx = np.zeros(imax)

print("")

for i in np.arange(1,imax+1):

    print("benchmarking i = ",i)
    
    indexx[i-1] = i
    m = 2**i
    n = 2**i

    # clean up old arrays 
    
    if 'A' in locals():
        del A
        del P1A,P2A
        del U, S,VH
        del UH,SI,V
        del Q, R, QI, RI

    # generate the matrix to be inverted
        
    A = np.random.random((m,n))

    # method 1; invert by SVD 
    
    tstart = time.time()
    U, S, VH = LA.svd(A);
    UH = np.asmatrix(U).H
    V  = np.asmatrix(VH).H
    
    SI = np.reciprocal(S)    
    #SI = 1.0/S           # equivalent to reciprocal
    
    P1A = np.dot(V,np.dot(np.diag(SI),UH))
    tstop = time.time()
    svtime[i-1] = tstop-tstart

    # method 2; invert by QR decomposition 
    
    tstart = time.time()
    Q, R = LA.qr(A);

    QI = M.transpose(Q)   # transpose faster than inversion
    #QI = LA.inv(Q)       # inversion slower than transpose

    RI = solve_triangular(R, np.identity(n)) # faster than LA.inv
    #RI = LA.inv(R)

    P2A = np.dot(RI,QI)
    tstop = time.time()
    qrtime[i-1] = tstop-tstart

### print to standard output ###

print("")
print("inversion time for 1024x1024 matrix:")
print("")
print("SV Decomposition: %8.5f seconds" % svtime[imax-1])
print("QR Decomposition: %8.5f seconds" % qrtime[imax-1])
print("")
if(svtime[imax-1] < qrtime[imax-1]):
    print("SV Decomposition is faster!")
else:
    print("QR Decomposition is faster!")
print("")
    
### plot the results ###

plt.figure(figsize=(8,6))

plt.scatter(indexx,svtime,label="SV decomposition")
plt.scatter(indexx,qrtime,label="QR decomposition")

plt.xlabel("$i$; matrix dimension = $2^i \\times 2^i$")
plt.ylabel("execution time (seconds)")
plt.yscale('log')
plt.axis([0.0,11.0,0.00001,10.0])
plt.legend(loc="upper left")
plt.grid()
plt.savefig("benchmark.png")
plt.show()

### bonus plot: ratio of SV/QR ###

if(True):

    rashio = svtime/qrtime

    plt.figure(figsize=(8,6))

    plt.scatter(indexx,rashio)

    plt.xlabel("$i$; matrix dimension = $2^i \\times 2^i$")
    plt.ylabel("$t_{SVD}/t_{QR}$")
    plt.yscale('linear')
    plt.axis([0.0,11.0,0.0,5.0])
    plt.grid()
    plt.show()


#
