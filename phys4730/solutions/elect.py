#!/usr/bin/python3

import numpy as np
import sys

###############################
def binomial_pdf_approx(p,n,k):
###############################

# calculates the gaussian approximation to a binomial
# probability of getting exactly k occurrences of some
# event (whose likelihood is p) in n independent trials.

  x = k-n*p 
  var = n*p*(1-p) 

  return np.exp(-x*x/(2.*var))/np.sqrt(2*np.pi*var) 

###################
### main script ###
###################

nvotes = 1000000       # number of votes cast

# make sure the user has entered a command line argument

if(len(sys.argv)!=2):
  print("please call elect.py with a single command line argument")
  exit()

# set the single-voter probability to the command line argument
  
pcan = float(sys.argv[1])

# tie: exactly nvotes/2 for candidate

ptie = binomial_pdf_approx(pcan,nvotes,0.5*nvotes)

# win: (nvotes/2 + 1) or more for candidate

pwin = 0.0
for ncan in range(int(0.5*nvotes+1),nvotes+1):
    pwin += binomial_pdf_approx(pcan,nvotes,ncan)

# display the results
    
print(" ")
print("probability p that voter chooses candidate: %8.6f" %(pcan))
print("probability of tie:                         %8.6f" %(ptie)) 
print("probability of win:                         %8.6f" %(pwin)) 
print(" ")


#
