import AeroProcess as ap
import matplotlib.pyplot as plt
import numpy as np
from LaTeXPy import latexify

"""
Plot of a two aircrafts paths, Montecarlo simulation 
of collition in function of the distance
"""

distance = 4 # distance(nmi)
Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 100 # numper of points in the trajectory
Time = 20

A1x, A1y = ap.AircraftTraj(Nsim, npoint, Time = Time)
A2x, A2y = ap.AircraftTraj(Nsim, npoint, Time = Time)
A2y += distance

latexify()
plt.figure()
plt.plot(A1x[0], A1y[0], 'b', lw = 2, label = 'Avion 1')
plt.plot(A2x[0], A2y[0], 'r', lw = 2, label = 'Avion 2')

plt.grid(True)
plt.legend()
plt.xlim([-20, 180])
plt.ylim([-4, distance + 4])
plt.xlabel("nmi")
plt.ylabel("nmi")
plt.plot(A1x[0,0], A1y[0,0], 'ko')
plt.plot(A1x[0,-1], A1y[0,-1], 'ko')
plt.plot(A2x[0,0], A2y[0,0], 'ko')
plt.plot(A2x[0,-1], A2y[0,-1], 'ko')
plt.savefig('Outputs/Script_5_1.pdf', bbox_inches='tight')

# Montecarlo method

currdist = (A1x - A2x) **2 + (A1y - A2y) ** 2
mindist = np.min(currdist, axis = 1)
mindist = np.sqrt(mindist)
timemin = np.argmin(currdist, axis = 1)

# Plot of the separation distance through time of one trajectory
plt.figure()
plt.grid(True)
plt.plot(np.linspace(0,Time, npoint), np.sqrt(currdist[0]))
#plt.title("Sep distance through time")
plt.xlabel("time (min)")
plt.ylabel("distance (nmi)")
plt.savefig('Outputs/Script_5_2.pdf', bbox_inches='tight')

# Probability of collition
prob = np.mean(mindist < 0.1)
var = np.var(mindist < 0.1)
erreur = 1.96*np.sqrt(var - prob**2)/np.sqrt(Nsim)
print("MC estimation")
print(prob)
print("Intervalle de conficance")
print([prob-erreur, prob+erreur])

ind = mindist < 0.1

# Hist plot of the number of collitions in function of the mindistance
# Conditioned to collition (mindist < 0.1)
plt.figure()
plt.hist(mindist[ind], bins = 4, range=(0, 0.1)) # Distribution of distances less than 0.1
plt.title("Density of min sep distance conditioned to collition")
plt.xlabel("min sep distance")
plt.ylabel("Number of times")
plt.savefig('Outputs/Script_5_3.pdf', bbox_inches='tight')


# Hist plot of the mindistance for Nsim simulations
plt.figure()
Long = np.max(mindist) - np.min(mindist)
Nbins = int(round(Nsim**(1./3.)*Long/3.49))
plt.hist(mindist, bins = Nbins, range=(0, distance))
plt.title("Density of min sep distance")
plt.xlabel("min sep distance")
plt.ylabel("Number of times")
plt.savefig('Outputs/Script_5_4.pdf', bbox_inches='tight')


# Plot of the distance through time for Nsim simulations
plt.figure()
plt.grid(True)
for i in range(10**3):
    plt.plot(np.linspace(0,Time, npoint), np.sqrt(currdist[i]))
#plt.title("Sep distance through time for several samples")
plt.xlabel("time (min)")
plt.ylabel("distance (nmi)")
plt.savefig('Outputs/Script_5_5.pdf', bbox_inches='tight')


# Hist plot of time of collition
# Conditioned to collition (mindist < 0.1)
plt.figure()
colindex = timemin[ind]
coltime = np.linspace(0,Time,npoint)[colindex]
Long = np.max(coltime) - np.min(coltime)
size = coltime.size
Nbins = int(round(size**(1./3.)*Long/3.49))
plt.hist(coltime, bins = Nbins, range=(0, Time)) # Distribution of distances less than 0.1
plt.title("Density of time of collition conditioned to collition")
plt.xlabel("Time")
plt.ylabel("Number of times")
plt.savefig('Outputs/Script_5_6.pdf', bbox_inches='tight')


# Hist plot of time of collition
# Conditioned to collition (mindist < 0.1)
plt.figure()
colindex = timemin[ind]
coll = A1x[ind, colindex]
Long = np.max(coll) - np.min(coll)
size = coll.size
Nbins = int(round(size**(1./3.)*Long/3.49))
plt.hist(coll, bins = 5, range=(0, Time*500/60)) # Distribution of distances less than 0.1
plt.title("Density of position of collition conditioned to collition")
plt.xlabel("Position")
plt.ylabel("Number of times")
plt.savefig('Outputs/Script_5_7.pdf', bbox_inches='tight')


plt.show()