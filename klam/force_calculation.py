import numpy as np
import numba as nb

@nb.njit
def calc_force_potential(particles, boxLength, sigma, epsilon, cutOfDistance):

    numberOfParticles = len(particles)
    forces = np.zeros((numberOfParticles, 2))
    potentialEnergy = 0.0

    for i in range(numberOfParticles):
        for j in range(i + 1, numberOfParticles):
            dx = particles[i, 0] - particles[j, 0]
            dy = particles[i, 1] - particles[j, 1]

            dx -= boxLength * np.round(dx / boxLength)
            dy -= boxLength * np.round(dy / boxLength)

            distanceSquared = dx**2 + dy**2
            distance = np.sqrt(distanceSquared)

            if distance < cutOfDistance:
                potentialEnergy += epsilon * ((sigma / distance)**2 - 1) * ((cutOfDistance / distance)**2 - 1)**2
                force = (2.0 * epsilon * (cutOfDistance**2 - distanceSquared) * (3.0 * sigma**2 * cutOfDistance**2 - (sigma**2 + 2.0 * cutOfDistance**2) * distanceSquared) / (distanceSquared**4))
                
                forces[i, 0] += force * dx
                forces[i, 1] += force * dy

                forces[j, 0] -= force * dx
                forces[j, 1] -= force * dy

    return forces, potentialEnergy