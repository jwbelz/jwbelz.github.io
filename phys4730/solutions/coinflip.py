#!/usr/bin/python3

import random
import numpy as np

###################
def headsortails():
###################
    
    return random.choice(['H','T'])

#################
def factorial(g):
#################
    
    if g > 100:
        print(g,"is too big!")
        sys.exit(0)
    else:
        fact = 1
        for i in range(1,g+1):
            fact = fact*i
    return fact

################################
def binomial_probability(p,N,k):
################################
    
    return factorial(N)/(factorial(k)*factorial(N-k))*pow(p,k)*pow(1.0-p,N-k)

########################
def binomial_error(p,N):
########################
    
    return np.sqrt(N*p*(1.0-p))

###############################
def binomial_pdf_approx(p,n,k):
###############################

#   calculates the gaussian approximation to a binomial
#   probability of getting exactly k occurrences of some
#   event (whose likelihood is p) in n independent trials.

    x = k-n*p 
    var = n*p*(1-p) 

    return np.exp(-x*x/(2.*var))/np.sqrt(2*np.pi*var) 

###################
### main script ###
###################

nflips = 10       # number of coin flips to simulate
ntries = 1000000  # number of times to simulate the flipping of nflips coins
p      = 0.5      # probability of heads with a fair coin
k      = 5        # expected number of successes

nheadstails = 0   # a counter, number of times nheads = ntails

for i in range(0,ntries):

    nheads = 0
    ntails = 0

    for j in range(0,nflips):

        result = headsortails()
        if(result == 'H'):
            nheads += 1
        else:
            ntails += 1

    if(nheads == ntails): nheadstails += 1

prob   = float(nheadstails)/float(ntries)
bprob  = binomial_probability(p,nflips,k)
bprobe = binomial_error(bprob,ntries)/float(ntries)
bprobg = binomial_pdf_approx(p,nflips,k)

print(" ")
print("observed frac nheads = ntails: %7.5f " % prob)
print("binomial prob nheads = ntails: %7.5f +- %7.5f " % (bprob,bprobe))
print("gaussian approximation:        %7.5f " % bprobg)
print(" ")

#
