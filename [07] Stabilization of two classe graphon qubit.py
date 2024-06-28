# -*- coding: utf-8 -*-
"""

@author: Sofiane Chalal (L2S)

Stabilization two classe graphon Belavkin equation.

"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qtp


dt = 0.001
steps = int(1/dt) # Nombres de pas du processus
Times = np.linspace(0., 1, steps) # Times
i = 1j #Imaginary number i 
'-----------------------------------------------'
'-----------------------------------------------'
'-----------------------------------------------'


" Matrix density input "
def density_matrix(x,y,z):
    i = 1j
    RHO = [[1+z, x - i*y], [x + i*y, 1 - z]]
    RHO = qtp.Qobj(RHO)
    RHO = 0.5*RHO
    return RHO
'-----------------------------------------------'
'-----------------------------------------------'
'-----------------------------------------------'


"Pauli Matrices "
sigx = qtp.sigmax()
sigy = qtp.sigmay()
sigz = qtp.sigmaz()

'-----------------------------------------------'
'-----------------------------------------------'
'-----------------------------------------------'

Hc = sigx # Laser control 
L =  sigz # Measurement channel
H =  sigz # Free Hamiltonian
N = 10 # Numbers of particles
rho0 = density_matrix(0, 0, 0) #Initialization

'-----------------------------------------------'
'-----------------------------------------------'
'-----------------------------------------------'
 

rhoC1 = density_matrix(0,0,1) #Target state for the first class qubit
rhoC2 = density_matrix(0,0,-1) #Target state for the second class qubit

'-----------------------------------------------'
'-----------------------------------------------'
'-----------------------------------------------'

" Pauli compenents for the two class particles "
x1, y1, z1 = np.zeros((N,steps)),np.zeros((N,steps)),np.zeros((N,steps))
x2, y2, z2 = np.zeros((N,steps)), np.zeros((N,steps)), np.zeros((N,steps))

'-----------------------------------------------'
'-----------------------------------------------'
'-----------------------------------------------'

"Fidelity distance between two class particles and their their stabilization target states "
F1, F2 = np.zeros(steps), np.zeros(steps)
F1[0] = qtp.fidelity(rho0,rhoC1)
F2[0] = qtp.fidelity(rho0,rhoC2)

'-----------------------------------------------'
'-----------------------------------------------'
'-----------------------------------------------'

"Initialization "
for n in range(N):
    x1[n][0] = (rho0*sigx).tr()
    y1[n][0] = (rho0*sigy).tr()
    z1[n][0] = (rho0*sigz).tr()
    x2[n][0] = (rho0*sigx).tr()
    y2[n][0] = (rho0*sigy).tr()
    z2[n][0] = (rho0*sigz).tr()
'-----------------------------------------------'
'-----------------------------------------------'
'-----------------------------------------------'

"Empirical measures "
mu1x, mu1y,mu1z = [], [], []
mu2x, mu2y,mu2z = [], [], []
'-----------------------------------------------'
'-----------------------------------------------'
'-----------------------------------------------'

mu1x.append(x1[0][0])
mu1y.append(y1[0][0])
mu1z.append(z1[0][0])

mu2x.append(x2[0][0])
mu2y.append(y2[0][0])
mu2z.append(z2[0][0])
'-----------------------------------------------'
'-----------------------------------------------'
'-----------------------------------------------'



    
def control(Hc,rho,rhoc):
    i = 1j
    u = -8*((i*((qtp.commutator(sigx,rho))*rhoc)).tr()) + 5*(1 - ((rho*rhoc).tr())**2)
    Hcu = -i*qtp.commutator(u*Hc,rho)
    
    return Hcu


def ham(H,rho):
    i = 1j
    dH =  -i*(qtp.commutator(H,rho))
    return dH

def deco(L,rho):
    dL = L*rho*(L.dag()) - (1/2)*(qtp.commutator(L.dag()*L,rho, "anti"))
    return dL



def mf(mx,my,mz,rho):
    i = 1j
    #a = 0.5*qtp.Qobj(([0,mx - i*my],[mx + i*my,0]))
    a = 0.5*qtp.Qobj(([1 - mz,0],[0,1+mz]))
    #a = 0.5*qtp.Qobj(([1+mz, -(mx - i*my)], [-(mx + i*my), 1 - mz]))
    mf = -i*(qtp.commutator(a,rho))
    
    return mf

def mes(L,rho):
    mes = L*rho + rho*L.dag() - (((L.dag()+L)*rho).tr())*rho

    return mes




'---------------------------------------------------------------------------'

for tt in range(steps-1):
    for n in range(N):
        rho1 = density_matrix(x1[n][tt], y1[n][tt], z1[n][tt])
        rho2 = density_matrix(x2[n][tt], y2[n][tt], z2[n][tt])
        lind1 = control(Hc,rho1,rhoC1) + ham(H,rho1) + deco(L,rho1) + mf(mu2x[tt],mu2y[tt],mu2z[tt],rho1)
        lind2 = control(Hc,rho2,rhoC2) + ham(H,rho2) + deco(L,rho2) + mf(mu1x[tt],mu1y[tt],mu1z[tt],rho2)
        rho1 = rho1 + lind1*dt + mes(L,rho1)*np.sqrt(dt)*np.random.normal(0,1) 
        rho2 = rho2 + lind2*dt + mes(L,rho2)*np.sqrt(dt)*np.random.normal(0,1)
        '------------------------------------------'
        x1[n][tt+1] = (rho1*sigx).tr()
        y1[n][tt+1] = (rho1*sigy).tr()
        z1[n][tt+1] = (rho1*sigz).tr()
        '-----------------------------------------'
        x2[n][tt+1] = (rho2*sigx).tr()
        y2[n][tt+1] = (rho2*sigy).tr()
        z2[n][tt+1] = (rho2*sigz).tr()
    '-----------------------------------------------'
    '-----------------------------------------------'
    '-----------------------------------------------'
    
    ax1 = 0
    ay1 = 0
    az1 = 0
    ax2 = 0
    ay2 = 0
    az2 = 0
    for n in range(N):
        ax1 = ax1 + x1[n][tt+1]
        ay1 = ay1 + y1[n][tt+1]
        az1 = az1 + z1[n][tt+1]
        ax2 = ax2 + x2[n][tt+1]
        ay2 = ay2 + y2[n][tt+1]
        az2 = az2 + z2[n][tt+1]
    mu1x.append(ax1/N)
    mu1y.append(ay1/N)
    mu1z.append(az1/N)
    mu2x.append(ax2/N)
    mu2y.append(ay2/N)
    mu2z.append(az2/N)
    
    mu1 = density_matrix(mu1x[tt],mu1y[tt],mu1z[tt])
    mu2 = density_matrix(mu2x[tt],mu2y[tt],mu2z[tt])
    
    '-----------------------------------------------'
    '-----------------------------------------------'
    '-----------------------------------------------'
    
    F1[tt+1] = qtp.fidelity(mu1,rhoC1)
    F2[tt+1] = qtp.fidelity(mu2,rhoC2)
    

'-----------------------------------------------------------------------'
'-----------------------------------------------------------------------'
'-----------------------------------------------------------------------'
'-----------------------------------------------------------------------'
'-----------------------------------------------------------------------'
'-----------------------------------------------------------------------'  


  
'*****************************************************'
fig, ax1 =  plt.subplots(1,1,figsize=(16,8))


ax1.plot(Times, x1[1][:], lw = 0.5, color = 'darkblue', label = 'x')
ax1.plot(Times, y1[1][:], lw = 0.5, color = 'darkred', label = 'y')
ax1.plot(Times, z1[1][:], lw = 0.5, color = 'darkgreen', label = 'z')


plt.title("Evolution first class particle")
plt.xlabel("Times")
plt.ylabel("Evolution")
plt.legend()
plt.grid()   
        
'*****************************************************'
fig, ax2 = plt.subplots(1,1,figsize =(16,8))

ax2.plot(Times, x2[1][:], lw = 0.5, color = 'darkblue', label = 'x')
ax2.plot(Times, y2[1][:], lw = 0.5, color = 'darkred', label = 'y')
ax2.plot(Times, z2[1][:], lw = 0.5, color = 'darkgreen', label = 'z')


plt.title("Evolution second class particle")
plt.xlabel("Times")
plt.ylabel("Evolution")
plt.legend()
plt.grid() 
    
'*****************************************************'
fig, ax3 = plt.subplots(1,1,figsize = (16,8))

ax3.plot(Times, F1[:], lw = 0.7, color = 'brown', label = 'first class')
ax3.plot(Times, F2[:], lw = 0.7, color = 'orange', label = 'second class')


plt.title("Fidelity")
plt.xlabel("Times")
plt.ylabel("Evolution")
plt.legend()
plt.grid() 
    