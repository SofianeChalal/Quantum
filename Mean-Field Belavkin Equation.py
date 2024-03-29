# -*- coding: utf-8 -*-
"""
@author: Sofiane Chalal

Return to basics : Mean-field Controlled Belavkin equation. 
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qtp

########################################################
font1 = {'family':'serif','color':'darkred','size':17}
font2 = {'family':'serif','color':'darkblue','size':11}
########################################################

dt = 0.001
steps = int(1/dt) # Nombres de pas du processus
Times = np.linspace(0., 1, steps) # Le temps physique

i = 1j #Imaginary number i 



'Matrix density input '

def density_matrix(x,y,z):
    i = 1j
    RHO = [[1+z, x - i*y], [x + i*y, 1 - z]]
    RHO = qtp.Qobj(RHO)
    RHO = 0.5*RHO
    return(RHO)

'--------------------'
sigx = qtp.sigmax()
sigy = qtp.sigmay()
sigz = qtp.sigmaz()
'--------------------'

Hc = sigx # Laser control 
L =   sigz # Measurement channel
H = sigz # Free Hamiltonian
N = 100 # Numbers of particles 
eta = 0.8 # Efficiency measurement 

'------------------------------------------------'
rho0 = density_matrix(1/4, -1/4, 0)
rhoC = density_matrix(0,0,1)
'------------------------------------------------'

'-----------------------------------------------'
xN = np.zeros((N,steps))
yN = np.zeros((N,steps))
zN = np.zeros((N,steps))
Fidelity = np.zeros((N,steps))
'-----------------------------------------------'
'-----------------------------------------'
"Initialization "
for n in range(N):
    xN[n][0] = (rho0*sigx).tr()
    yN[n][0] = (rho0*sigy).tr()
    zN[n][0] = (rho0*sigz).tr()
    Fidelity[n][0] = qtp.fidelity(rho0,rhoC)
'-----------------------------------------'

print(Fidelity[0][0])


'-----------------------------------------'
"Empirical measures "
mu_x = []
mu_y = []
mu_z = []
'-----------------------------------------'


'----------------------------------------'
mu_x.append(xN[0][0])
mu_y.append(yN[0][0])
mu_z.append(zN[0][0])
'-----------------------------------------'

def drift_energy(H,rho):
    dH =  -i*(qtp.commutator(H,rho))
    return dH

def drift_decoherence(L,rho):
    dL = L*rho*(L.dag()) - (1/2)*(qtp.commutator(L.dag()*L,rho, "anti"))
    return dL


def U_control(direction : str, rho, rho_c, alpha : 7.61, beta : 5):
    i = 1j
    if (direction == 'x'):
        F = sigx
    elif (direction == 'y'):
        F = sigy
    elif (direction == 'z'):
        F = sigz 
    
    uc = -alpha*((i*((qtp.commutator(F,rho))*rho_c)).tr()) + beta*(1 - (rho*rho_c).tr())
    
    return uc
  
def control(Hc,rho):
    u = U_control('x',rho,sigz,7.61,5)
    Hcu = -i*qtp.commutator(u*Hc,rho)
    return Hcu


def mf_inter(mx,my,mz,rho):
    a = qtp.Qobj(([0,mx - i*my],[mx + i*my,0]))
    mf = -i*(qtp.commutator(a,rho))
    return mf

def diffusion_term(L,rho):
    diffusion = L*rho + rho*L.dag() - (((L.dag()+L)*rho).tr())*rho

    return diffusion

'---------------------------------------------------------------------------'

for tt in range(steps-1):
    dW = np.sqrt(dt)*np.random.normal(0,1,N)
    for n in range(N):
        rho = density_matrix(xN[n][tt], yN[n][tt], zN[n][tt])
        Lindblad = drift_energy(H, rho) + drift_decoherence(L, rho) + mf_inter(mu_x[tt],mu_y[tt],mu_z[tt],rho)
        rho = rho + np.sqrt(eta)*diffusion_term(L,rho)*dW[n] + Lindblad*dt + control(Hc,rho)*dt
        xN[n][tt+1] = (rho*sigx).tr()
        yN[n][tt+1] = (rho*sigy).tr()
        zN[n][tt+1] = (rho*sigz).tr()
        Fidelity[n][tt+1] = qtp.fidelity(rho,rhoC)
    '--------------------------------------'
    ax = 0
    ay = 0
    az = 0
    '--------------------------------------'
    for n in range(N):
        ax = ax + xN[n][tt+1]
        ay = ay + yN[n][tt+1]
        az = az + zN[n][tt+1]
    mu_x.append(ax/N)
    mu_y.append(ay/N)
    mu_z.append(az/N)
    '--------------------------------------'


Mean_Fidelity = np.zeros(steps)
for tt in range(steps):
    for n in range(N):
        Mean_Fidelity[tt] = Mean_Fidelity[tt] + Fidelity[n][tt]

Mean_Fidelity = Mean_Fidelity/N
        
        
# Z-direction of spin 
fig, axZ =  plt.subplots(1,1,figsize=(16,8))


for n in range(0,N):
    if (zN[n][steps-3] > 0.8):
        axZ.plot(Times, zN[n][:], lw = 0.5, color = 'darkblue')
    if (zN[n][steps-3] < -0.8):
        axZ.plot(Times, zN[n][:], lw = 0.5, color = 'darkred')
axZ.plot(Times, mu_z, lw = 3, color = 'red')

plt.title("Evolution of z", fontdict = font1)
plt.xlabel("Times", fontdict = font2)
plt.ylabel("Evolution", fontdict = font2)
plt.legend()
plt.grid()
'----------------------------------------------------'

'----------------------------------------------------------------------'
# Fidelity function evolution
fig, axFi = plt.subplots(1,1, figsize=(16,8))

for n in range(0,N):
    axFi.plot(Times, Fidelity[n][:], lw = 0.5, color = 'darkgreen')
axFi.plot(Times, Mean_Fidelity, lw = 3, color = 'red' )
plt.title("Fidelity ", fontdict = font1)
plt.xlabel("Times", fontdict = font2)
plt.ylabel("Evolution", fontdict = font2)
plt.legend()
plt.grid()
'-----------------------------------------------------------------------'
        

'-----------------------------------------------------------------------'
#The evolution of empirical measure
fig, axEmp = plt.subplots(1,1, figsize=(16,8))


axEmp.plot(Times, mu_x, lw = 0.6, color = 'darkgreen')
axEmp.plot(Times, mu_y, lw = 0.6, color = 'darkred')
axEmp.plot(Times, mu_z, lw = 0.6, color = 'darkblue')
plt.title('Empirical Measure', fontdict = font1)
plt.xlabel('Times', fontdict = font2)
plt.ylabel("Evolution", fontdict = font2)
plt.legend()
plt.grid()
'----------------------------------------------------------------------'
