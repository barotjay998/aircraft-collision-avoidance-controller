import AeroProcess as ap
import matplotlib.pyplot as plt
import numpy as np

"""
Plot of a  long trajectory between waypoints
at speed 500/60 nmi/min.
"""
Nsim = 1
npoint = 100
v = 500/60
Time = 20

tet = 90
tet1 = 30
tet2 = 30

a1, c1 = ap.TrajOblique([0,0], tet, Nsim, npoint, Time=Time, v =v)
a2, c2 = ap.TrajOblique([a1[0,-1], c1[0,-1]],tet1, Nsim, npoint, Time=Time, v =v)
a3, c3 = ap.TrajOblique([a2[0,-1], c2[0,-1]], tet2, Nsim, npoint, Time=Time, v =v)

Xa = np.concatenate((a1,a2,a3), axis=1)
Xc = np.concatenate((c1,c2,c3), axis=1)

plt.figure()
plt.plot(Xa[0], Xc[0], 'b', lw = 2)

plt.grid(True)
plt.xlabel("nmi")
plt.ylabel("nmi")
dist = v * Time

p, q = np.cos(np.radians(tet)), np.sin(np.radians(tet))
p1, q1 = np.cos(np.radians(tet1)), np.sin(np.radians(tet1))
p2, q2 = np.cos(np.radians(tet2)), np.sin(np.radians(tet2))

plt.plot(Xa[0,0], Xc[0,0], 'ro', markersize=10)
plt.plot(Xa[0,0] + dist*p, Xc[0,0] + dist*q, 'ro', markersize=10)
plt.plot(Xa[0,0]+ dist*(p+p1), Xc[0,0] + dist*(q+q1), 'ro', markersize=10)
plt.plot(Xa[0,0]+ dist*(p+p1+p2), Xc[0,0] + dist*(q+q1+q2), 'ro', markersize=10)

plt.savefig('Outputs/Script_4_1.pdf', bbox_inches='tight')
plt.show()