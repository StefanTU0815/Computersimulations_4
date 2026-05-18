from initialization import ParticleSystem, calc_system_temperature
from force_calculation import calc_force_potential
from particle_dynamics import velocity_verlet_step

import numpy as np
import matplotlib.pyplot as plt


targetTemperatures = [1.0, 5.0]

numberOfParticles = 100
packingFraction = 0.3

sigma = 1.0
epsilon = 1.0
cutOfDistance = 2.0 * sigma
mass = 1.0

dt = 0.005
numberOfSteps = 5000
dimensions = 2

delta = 0.05
thermostatInterval = 10
thermostat = True

plt.figure(figsize=(10, 6))

for targetTemperature in targetTemperatures:

    particleSystem = ParticleSystem(numberOfParticles, packingFraction, sigma, epsilon, cutOfDistance, targetTemperature, mass, dimensions)
    particles = particleSystem.initialize_particles()
    velocities = particleSystem.initialize_velocities()
    forces, potentialEnergy = calc_force_potential(particles, particleSystem.boxLength, sigma, epsilon, cutOfDistance)

    times = np.zeros(numberOfSteps + 1)
    temperatures = np.zeros(numberOfSteps + 1)

    temperatures[0] = calc_system_temperature(velocities, mass, dimensions)

    for step in range(1 , numberOfSteps + 1):
        particles, velocities, forces, potentialEnergy = velocity_verlet_step(particles, velocities, forces, mass, dt, particleSystem.boxLength, sigma, epsilon, cutOfDistance, thermostat, targetTemperature, dimensions, delta, thermostatInterval, step)
        times[step] = step * dt
        temperatures[step] = calc_system_temperature(velocities, mass, dimensions)

    plt.plot(times, temperatures, label=f'Target Temperature: {targetTemperature}')
    plt.axhline(targetTemperature, color='k')

plt.xlabel("time")
plt.ylabel("temperature")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()