import numpy as np
import numba as nb
from initialization import calc_system_kinetic_energy, calc_system_temperature
from force_calculation import calc_force_potential

@nb.njit
def heyes_tempostat_step(velocities, targetTemperature, dimensions, delta, mass):
    lnZ = np.random.uniform(-delta, delta)
    z = np.exp(lnZ)
    Nf = velocities.shape[0] * dimensions - dimensions - 1
    kineticEnergy = calc_system_kinetic_energy(velocities, mass)
    exponent = -kineticEnergy * (z**2 - 1) / targetTemperature
    acceptanceProbability = min(1.0, (z**Nf) * np.exp(exponent))

    if np.random.rand() < acceptanceProbability:
        for i in range(velocities.shape[0]):
            for d in range(dimensions):
                velocities[i, d] *= z

    return velocities

@nb.njit
def velocity_verlet_step(particles, velocities, forces, mass, dt, boxLength, sigma, epsilon, cutOfDistance, therostat=False, targetTemperature=1.0, dimensions=2, delta=0.1, thermostatInterval = 10, step=0):
    velocities += 0.5 * forces / mass * dt

    if therostat:
        if step % thermostatInterval == 0:
            velocities = heyes_tempostat_step(velocities, targetTemperature, dimensions, delta, mass)

    particles += velocities * dt
    particles = particles % boxLength
    forces, potentialEnergy = calc_force_potential(particles, boxLength, sigma, epsilon, cutOfDistance)
    velocities += 0.5 * forces / mass * dt

    return particles, velocities, forces, potentialEnergy

@nb.njit
def velocity_verlet_step_unwrapped(particles, unwrappedPositions, velocities, forces, mass, dt, boxLength, sigma, epsilon, cutOfDistance, thermostat=False, targetTemperature=1.0, dimensions=2, delta=0.1, thermostatInterval=10, step=0):
    velocities += 0.5 * forces / mass * dt

    if thermostat:
        if step % thermostatInterval == 0:
            velocities = heyes_tempostat_step(velocities, targetTemperature, dimensions, delta, mass)

    unwrappedPositions += velocities * dt
    particles = unwrappedPositions % boxLength
    forces, potentialEnergy = calc_force_potential(particles, boxLength, sigma, epsilon, cutOfDistance)
    velocities += 0.5 * forces / mass * dt

    return particles, unwrappedPositions, velocities, forces, potentialEnergy
