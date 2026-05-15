import numpy as np
import numba as nb

@nb.njit
def calc_radial_distribution(particles, numberOfRValues, boxLength):
    numberOfParticles = particles.shape[0]
    g = np.zeros(numberOfRValues)
    rValues = np.zeros(numberOfRValues)

    rMax = boxLength / 2.0
    dr = rMax / numberOfRValues

    for k in range(numberOfRValues):
        rValues[k] = (k + 0.5) * dr

    for i in range(numberOfParticles):
        for j in range(i + 1, numberOfParticles):
            dx = particles[i, 0] - particles[j, 0]
            dy = particles[i, 1] - particles[j, 1]

            dx -= boxLength * np.round(dx / boxLength)
            dy -= boxLength * np.round(dy / boxLength)

            distance = np.sqrt(dx**2 + dy**2)

            if distance < rMax:
                binIndex = int(distance / dr)

                if binIndex < numberOfRValues:
                    r = rValues[binIndex]
                    g[binIndex] += 1.0 / (2.0 * np.pi * r * dr)

    g *= 2.0 * boxLength**2 / numberOfParticles**2

    return rValues, g