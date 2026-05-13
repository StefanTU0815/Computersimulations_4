# %% [markdown]
# %%
#1a
import numpy as np
import matplotlib.pyplot as plt 

#Constants
sigma = 1.0
rc = 2.5*sigma
epsilon = 1.0
phi = 0.3

#Parameter
N = 100

#function to generate positions with the given tensity -> this variant is very slow 
def generate_positions(N, phi, sigma, factor = 1.0, max_trials=100000):
    positions = []
    cnt = 0

    #calculate A for the chosen N and sigma
    A = (N*np.pi*sigma**2) / phi 

    #quadradic field with sidelenght L
    L = np.sqrt(A)

    #calc minimal distance between two moleculecenters
    min_distance = 2* sigma * factor

    #try as long as their are N molecules or max_tails is reached
    while len(positions) < N and cnt < max_trials:
        p = np.array([np.random.uniform(0, L),
                      np.random.uniform(0, L)])

        # check if their is an overlap
        ok = True
        for q in positions:
            if np.linalg.norm(p - q) < min_distance:
                ok = False
                break

        if ok:
            positions.append(p)

        cnt += 1

    # if the number of paricles cannot be reached within the max 
    # nuber of trails stopp with the error 
    if len(positions) < N:
        raise RuntimeError("density not reachable.")

    return np.array(positions), L


#function to generate positions with the given tensity, vectorized 
def generate_positions_v(N, phi, sigma, factor=1.0, max_trials=100000):
    #create empty arry
    positions = np.empty((2, 0))
    cnt = 0

    # calc A / L  for the chosen density
    A = (N * np.pi * sigma**2) / phi
    L = np.sqrt(A)

    # calc minimal distances between the molecule centers
    min_distance = 2 * sigma * factor

    # create N positions
    while positions.shape[1] < N and cnt < max_trials:
        cnt += 1

        #random guess of new positons
        p = np.array([
            np.random.uniform(0, L),
            np.random.uniform(0, L)
        ])

        # first iteration ?
        if positions.shape[1] == 0:
            positions = np.column_stack((positions, p))
            continue

        # deltas from every existing particel to the new guess
        delta = positions - p[:,np.newaxis]

        # calc distance
        distances = np.sqrt(delta[0, :]**2 + delta[1, :]**2)

        # is the particle in the critical region of another particle? 
        if np.all(distances >= min_distance):
            positions = np.column_stack((positions, p))

    # if N not reached -> error message
    if positions.shape[1] < N:
        raise RuntimeError("density not reachable.")
    
    # transpose 
    positions = positions.T
    return positions, L


#positions, L = generate_positions(N, phi, sigma, 1.0, 1000000)
positions, L = generate_positions_v(N, phi, sigma, 1.0, 1000000)

print("Box length L =", L)
print("Generated positions:")
print(positions)
print(positions.shape)



# %%
#Testfeld
#plot particles
fig, ax = plt.subplots(figsize=(6, 6))

for x, y in positions:
    circle = plt.Circle((x, y), sigma, fill=False)
    ax.add_patch(circle)

ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("2D non-overlapping molecules")
plt.show()

# %% [markdown]
# %%
#1b
# function to generate the velocities
def generate_velocities(N, T, k_B, m):
    # for corresbondig to a Temp use a sigma_v(T)
    # Äquipartitionstheorem 1/2 kb T per dimension
    sigma_v = np.sqrt(k_B*T/m)
    # create velocities
    v = np.random.normal(0.0, sigma_v, size=(N, 2))
    # subtract center-of-mass velocity
    v_cm = np.mean(v, axis=0)
    v_cm_corr = v - v_cm

    return v, v_cm_corr 

# check the function (vanishes v of des center of mass)
N = 100
m =1.0
k_B = 1.0
T = 1.0

v, v_cm_corr = generate_velocities(N, T, k_B, m)
print("Number of Values", np.shape(v))
print("Means of  direction (without COM correction):", np.mean(v, axis=0))
print("Means of  direction (with COM correction):", np.mean(v_cm_corr, axis=0))
#print(positions)


# %% [markdown]
# %%
#2a
def pair_force(vec_rij, epsilon, sigma, rc):
    # absolutvalue of the vector rij
    r = np.sqrt(np.dot(vec_rij,vec_rij))

    # chec if inside the cutoff distance (avoid division 0 )
    if r >= rc or r < 1e-48:
        return np.zeros(2)
    
    # calculate a force
    A = (sigma**2 / r**2) -1.0
    B = (rc**2 / r**2) -1.0
    
    f = vec_rij * (2*epsilon/r**4) * (sigma**2 * B**2 + 2 * A * B * rc**2)
    return f

# test the function
#rc = 10
#i = 0
#j = 6
#rij = positions[i]-positions[j]
#r = np.linalg.norm(rij)
#f = pair_force(rij, epsilon, sigma, rc)
#print(rij)
#print(r)
#print(f)
#print(positions)

# calculate total forces
def total_forces(positions, L, epsilon, sigma, rc):
    N = len(positions)
    forces = np.zeros_like(positions)
    U_tot = 0
    # seite 79 im scirpt!
    for i in range(N - 1):
        for j in range(i + 1, N):
            rij = positions[i] - positions[j]
            # this is to have periodic poundary conditions
            rij -= L * np.round(rij / L)
            fij  = pair_force(rij, epsilon, sigma, rc)
            # add force to i
            forces[i] += fij
            # same force accts with opposite sign to j
            forces[j] -= fij

    return forces

# Test of te funciton
rc = 2.5
f_tot = total_forces(positions, L, epsilon, sigma, rc)
idx = np.where(np.linalg.norm(f_tot, axis=1) != 0)[0]
print(L)
print(f_tot[idx])


# %% [markdown]
# %%
#2b
def pair_force_potential_Energy(vec_rij, epsilon, sigma, rc):
    # absolutvalue of the vector rij
    r = np.sqrt(np.dot(vec_rij,vec_rij))

    # chec if inside the cutoff distance (avoid division 0 )
    if r >= rc or r < 1e-48:
        return np.zeros(2)
    
    # calculate a force
    A = (sigma**2 / r**2) -1.0
    B = (rc**2 / r**2) -1.0
    f = vec_rij * (2*epsilon/r**4) * (sigma**2 * B**2 + 2 * A * B * rc**2)

    # calc E Pot
    U = epsilon*A*B**2
    return f, U

# sum up forces and energys 
def total_forces_Energy(positions, L, epsilon, sigma, rc):
    N = len(positions)
    forces = np.zeros_like(positions)
    U_tot = 0
    # seite 79 im scirpt!
    for i in range(N - 1):
        for j in range(i + 1, N):
            rij = positions[i] - positions[j]
            # this is to have periodic poundary conditions
            rij -= L * np.round(rij / L)
            fij, Uij  = pair_force_potential_Energy(rij, epsilon, sigma, rc)
            # add force to i
            forces[i] += fij
            # same force accts with opposite sign to j
            forces[j] -= fij
            # summ up potential Energy
            U_tot += Uij
    return forces, U_tot

# Test of te funciton
rc = 2.5
f_tot, U_tot = total_forces_Energy(positions, L, epsilon, sigma, rc)
idx = np.where(np.linalg.norm(f_tot, axis=1) != 0)[0]
print(L)
print(f_tot[idx])
print(U_tot)


# %% [markdown]
# %%
#2c
import time
N_values = [50,100,150,200,250,400,600] 
times = []
repeats = 2 # per N_value

for N in N_values:
    positions, L = generate_positions_v(N, phi, sigma, 1.0, 1000000)
    # list for times with same N to calc mean
    t_list =[]
    # do this for the number of repeats
    for _ in range(repeats):
        t0 = time.perf_counter()
        f  = total_forces(positions, L, epsilon, sigma, rc)
        t1 = time.perf_counter()
        t_list.append(t1-t0)
    # calc mean of all times for same n
    times.append(np.mean(t_list)) 

# crate a fit to show its quatratic 
times = np.array(times)
N_values_plot = np.array(N_values)
coeff = np.polyfit(N_values_plot, times, 2)
a, b, c = coeff
N_fit = np.linspace(N_values_plot.min(), N_values_plot.max(), 300)
t_fit = a * N_fit**2 + b * N_fit + c

plt.plot(N_values_plot, times, 'o', label='needed time')
plt.plot(N_fit, t_fit, '-', label='quadratic fit')
plt.xlabel('N / 1')
plt.ylabel('runtime / s')
plt.legend()
plt.show()

print("Fit: t(N) = a*N^2 + b*N + c")
print("a =", a)
print("b =", b)
print("c =", c)


# %% [markdown]
# %%
#3a
# no code required

# %% [markdown]
# %% [markdown]
# %%
#3b
def velocity_verlet(pos, vel, L, n_steps, dt, epsilon, sigma, rc, m):
    # use N of pos
    N = len(pos)

    # calculate forces and Potential energy
    forces, U = total_forces_Energy(pos, L, epsilon=epsilon, sigma=sigma, rc=rc)

    # initalize arrays for energys
    E_kin_hist = np.zeros(n_steps)
    E_pot_hist = np.zeros(n_steps)
    E_tot_hist = np.zeros(n_steps)
    T_hist = np.zeros(n_steps)
    pos_hist = np.zeros((n_steps, N, 2))


    for i in range(n_steps):
        # calc fist velocity halfstep
        vel_half = vel + 0.5 * forces / m * dt

        # Full-step position
        pos = pos + vel_half * dt

        # Pos Modulo L is used for periodic boundary conditions
        pos = pos % L

        # New forces forces after step
        forces_new, U = total_forces_Energy(pos, L, epsilon=epsilon, sigma=sigma, rc=rc)

        # Full-step velocity
        vel = vel_half + 0.5 * forces_new / m * dt

        # Update forces
        forces = forces_new

        # calc kinetic enerty
        E_kin = 0.5 * m * np.sum(vel**2)

        # Temperature in 2D
        # degrees of freedom = 2N - 2
        dof = 2 * N - 2
        T = 2 * E_kin / dof

        E_kin_hist[i] = E_kin
        E_pot_hist[i] = U
        E_tot_hist[i] = E_kin + U
        T_hist[i] = T

        pos_hist[i] = pos
        print(i, "/",n_steps)
    return pos, vel, pos_hist, E_kin_hist, E_pot_hist, E_tot_hist, T_hist

# %% [markdown]
# %%
#3c
# Example parameters
N = 100
sigma = 1.0
epsilon = 1.0
m = 1.0
rc = 2.5
phi = 0.3
dt = 0.001
n_steps = 1000

positions, L = generate_positions_v(N, phi, sigma, 1.0, 1000000)
_, v = generate_velocities(N, 1, 1, m)

pos, vel, pos_hist, E_kin_hist, E_pot_hist, E_tot_hist, T_hist = velocity_verlet(
    positions,v, L,
    n_steps=n_steps,
    dt=dt,
    epsilon=epsilon,
    sigma=sigma,
    rc=rc,
    m=m
)

time = np.arange(n_steps) * dt

plt.figure()
plt.plot(time, E_tot_hist, label="E_tot")
plt.xlabel("time")
plt.ylabel("energy")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(time, E_kin_hist, label="E_kin")
plt.plot(time, E_pot_hist, label="E_pot")
plt.plot(time, E_tot_hist, label="E_tot")
plt.xlabel("time")
plt.ylabel("energy")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(time, T_hist)
plt.xlabel("time")
plt.ylabel("temperature")
plt.grid()
plt.show()


