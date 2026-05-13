import numpy as np
import matplotlib.pyplot as plt

# This is a basic MD algorithm to simulate N particles in a two-dimensional
# box of fixed dimensions. Particles interact via the Wang-Frenkel potential.
# Mitigating measures to speed up the force calculation have not been implemented.
# DJ Bonthuis, Graz, May 2026

# packing fraction 
phi = 0.7e0

# box size
L = 7e0
box = np.array([L,L])

# parameters of the interaction potential
eps = 1.0e0
sig = 1.0e0

# initial temperature (thermal energy in units kBT)
temp_0 = 1e0

# number of particles
# here we use a definition of phi that is different from the exercise sheet:
# phi = N pi (sig/2)^2 / L^2
# because the radius of the particles is better approximated by sig/2
Nx = round(np.sqrt(phi/np.pi)*L*2/sig)
Ny = round(np.sqrt(phi/np.pi)*L*2/sig)
N = Nx*Ny 

# particle mass
# (double array in order to work easily with the force vector)
m = np.ones((N,2))

# number of steps and time step (one per cent of the characteristic time)
N_steps = 1000
dt = 0.01e0 * sig*np.sqrt(np.min(m)/eps)
print("simulating %i particles for %i steps with a time step of %8.5f" % (N,N_steps,dt))


def initialize(N,temp_0):

  # regular distribution to keep the energy low
  x = np.zeros((N,2))
  for i in range(Nx):
    for j in range(Ny):
      x[i*Nx+j,0] = i*L/Nx - L/2e0
      x[i*Nx+j,1] = j*L/Ny - L/2e0

  # Gaussian distribution of the velocity
  U1 = np.random.random(N)
  U2 = np.random.random(N)
  v = np.zeros((N,2))
  v[:,0] = np.sqrt(-2e0*np.log(U1))*np.cos(U2*2e0*np.pi)
  v[:,1] = np.sqrt(-2e0*np.log(U1))*np.sin(U2*2e0*np.pi)

  # remove collective velocity
  v = v-np.mean(v,axis=0)

  # set the initial temperature
  Nf = 2*N - (2+1)
  temp_t = np.sum(m[:,0]*(v[:,0]**2+v[:,1]**2))/Nf
  v = v * np.sqrt(temp_0/temp_t)

  return x, v


def forces(x):

  N = len(x)
  f = np.zeros((N,2))
  U = 0e0
  for i in range(N):
    for j in range(i+1,N):
      rij = x[i,:]-x[j,:]
      rij = rij - np.round(rij/box)*box
      rij2 = np.dot(rij,rij)

      # Wang-Frenkel using rc = 2*sig
      if (rij2 < 4e0*sig**2):
        sig3 = sig**3
        f[i,:] += rij * eps * 6e0*(sig*rij2-4e0*sig3)*(3e0*sig*rij2-4e0*sig3)/rij2**4
        f[j,:] -= rij * eps * 6e0*(sig*rij2-4e0*sig3)*(3e0*sig*rij2-4e0*sig3)/rij2**4

        # calculate the total energy
        U += eps * (-rij2**3 + 9e0*sig**2*rij2**2 - 24e0*sig**4*rij2 + 16e0*sig**6)/rij2**3

  return f, U


def mdrun(N_steps,x,v,f):
  U = np.zeros(N_steps)
  K = np.zeros(N_steps)
  for i in range(N_steps):

    # velocity verlet
    v = v + f * dt/(2e0*m)
    x = x + v * dt
    f, U[i] = forces(x)
    v = v + f * dt/(2e0*m)

    K[i] = np.sum(m[:,0]*(v[:,0]**2+v[:,1]**2)/2e0)

  return U, K


x, v = initialize(N, temp_0)
f, dummy = forces(x)

U, K = mdrun(N_steps,x,v,f)

E = K+U
time = np.arange(N_steps)*dt
plt.plot(time,E,label='total')
plt.plot(time,K,label='kinetic')
plt.plot(time,U,label='potential')
# note that the unit of time depends on the units of mass m and length l as t = l sqrt(m/kBT)
# since we have not defined units of length and mass, our time is measured in arbitrary units
plt.xlabel("time (a.u.)")
plt.ylabel("energy (kBT)")
plt.legend()
plt.show()

