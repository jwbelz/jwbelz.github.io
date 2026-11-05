#!/usr/bin/python3

### Generalized Monte Hall Problem.              ###
### Write Monte Carlo to prove Bayesian result   ###
### for arbitrary number of doors $N$.           ###
###                                              ###
### Instructor solution for Lab20, Exercise 1    ###
### PHYS 4730 Fall 2026                          ###

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib
import sys
matplotlib.rcParams.update({'font.size': 16})

### Specify the number of doors ###

N = 3

### The user should have specified the #trials on the command line ###

if(len(sys.argv)>1):
    ntrials = int(sys.argv[1])
else:
    print("you need to enter the number of trials you want")
    exit()

### conduct the trials ###

switch_win  = 0
switch_lose = 0

for i in np.arange(0,ntrials):

    # hide the car behind a door
    
    prizedoor = random.randint(low=1,high=N+1,size=1,dtype=int)[0]
    
    # the player always, without losing generality, chooses door 1
    
    playerchoose = 1

    # reveal N-2 of remaining doors NOT hiding a car
    
    if(prizedoor==1):
        unrevealeddoor = random.randint(low=2,high=N+1,size=1,dtype=int)[0]
    else:
        unrevealeddoor = prizedoor
    
    # the player switches doors. Does the player win or lose the car?
    
    playerdoor = unrevealeddoor
    
    if(playerdoor==prizedoor):
        switch_win  += 1
    else:
        switch_lose += 1
        
### compute win fraction and binomial error ###

winfrac = switch_win/ntrials
winerr  = np.sqrt(switch_win*(1.0-switch_win/ntrials))/ntrials

theanswer = "No!"
if(switch_win > switch_lose):
    theanswer = "Yes!"
    
### output the results ###

print("")
print("the number of doors  = %5d" % N)
print("the number of trials = %5d" % ntrials)
print("")
print("the player wins by switching:     %5d" % switch_win)
print("the player wins by not switching: %5d" % switch_lose)
print("")
print("the probability of the player winning by switching is %7.5f \u00b1 %7.5f" % (winfrac,winerr))
print("")
print("should the player switch? %5s " % theanswer)
print("")

#
