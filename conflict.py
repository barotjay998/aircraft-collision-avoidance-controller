# -*- coding: utf-8 -*-
"""
Estimation via Monte carlo the conflict probability 
in function of the number of simulations

@author: felipe
"""

import AeroProcess as ap
import numpy as np


tet = 90 # Angle between trajectories
v = 500/60
Time = 20
# d1 and d2 set to zero makes the worst case
d1 = 0 # Distance from midpoint 1 traj
d2 = 0 # Distance from midpoint 1 traj
dist1 = Time * v / 2 - np.cos(np.radians(tet)) * Time * v / 2 # distance(nmi)
dist2 = -np.sin(np.radians(tet)) * Time * v / 2 # distance(nmi)

Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 100 # numper of points in the trajectory

A1x, A1y = ap.AircraftTraj(Nsim, npoint, Time = Time)
A2x, A2y = ap.TrajOblique([dist1,dist2], tet, Nsim, npoint, Time=Time)


# Montecarlo method

currdist = (A1x - A2x) **2 + (A1y - A2y) ** 2
mindist = np.min(currdist, axis = 1)
mindist = np.sqrt(mindist)
timemin = np.argmin(currdist, axis = 1)


# Probability of collition
prob = np.mean(mindist < 0.1)
var = np.var(mindist < 0.1)
erreur = 1.96*np.sqrt(var - prob**2)/np.sqrt(Nsim)
print("MC estimation")
print(prob)
print([prob-erreur, prob+erreur])