# Yukawa potential

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import bisect

def Veff(r):                    # effective potential
    return (L*(L+1)/(2*mass*r)-np.exp(-0.1*r))/r
    
def f(r):                       # Sch eqn in Numerov form
    return 2*mass*(E-Veff(r))
    
def numerov(f, u, n, x, h):     # Numerov integrator for $u''+f(x)u=0$
    nodes, c = 0, h*h/12.       # given $[u_0,u_1]$, return $[u_0,u_1,...,u_{n+1}]$
    f0, f1 = 0., f(x+h)
    for i in range(n):
        x += h
        f2 = f(x+h)             # Numerov method below, 
        u.append((2*(1-5*c*f1)*u[i+1] - (1+c*f0)*u[i])/(1+c*f2))  
        f0, f1 = f1, f2
        if (u[-1]*u[-2] < 0.1): nodes += 1
    return u, nodes             # return u, nodes
    
def shoot(En):
    global E                    # E needed in f(r)
    E, c, xm = En, (h*h)/6., xL + M*h
    wfup, nup = numerov(f, [0,.01], M, xL, h)
    wfdn, ndn = numerov(f, [0,.01], N-M, xR, -h)     # $f'$ from 
    dup = ((1+c*f(xm+h))*wfup[-1] - (1+c*f(xm-h))*wfup[-3])/(h+h)
    ddn = ((1+c*f(xm+h))*wfdn[-3] - (1+c*f(xm-h))*wfdn[-1])/(h+h)
    return dup*wfdn[-2] - wfup[-2]*ddn

xL, xR, N = 0., 120., 2200                  # limits, intervals
xa=np.linspace(xL, xR, N+1)
h, mass = (xR-xL)/N, 1.0                    # step size, mass
Lmax, EL, M = 1, [], 100                    # M = matching point


L=0
S='E%f*27.211 ev'
n, Ea, dE = L+1,[], 0.001  
Estart = -.5/np.arange(1, Lmax+1)**2-0.1,    # $\sim -1/2n^2$
E1=Estart[L]
# sweep E range for each L
while (E1 < -dE):
    E1 += dE
    if (shoot(E1)*shoot(E1 + dE) > 0): continue
    E = bisect(shoot, E1, E1 + dE)
    Ea.append(E)
    wfup, nup = numerov(f, [0,.1], M-1, xL, h)      # calc wf
    wfdn, ndn = numerov(f, [0,.1], N-M-1, xR, -h)
    psix = np.concatenate((wfup[:-1], wfdn[::-1]))
    psix[M:] *= wfup[-1]/wfdn[-1]                   # match
    plt.plot(xa, psix, label=S%E )
    plt.legend()
    print ('nodes, n,l,E=', nup+ndn, n, L, E*27.211)
    n += 1
    
plt.show()









