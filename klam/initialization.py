import numpy as np
import numba as nb
from numba.experimental import jitclass
from numba import float64, int64

@nb.njit
def calc_system_temperature(velocities, mass, dimensions):
    Nf = velocities.shape[0] * dimensions - dimensions - 1
    return mass * np.sum(velocities**2) / Nf

@nb.njit
def calc_system_kinetic_energy(velocities, mass):
    return 0.5 * mass * np.sum(velocities**2)


spec_particle_system = [
    ('numberOfParticles', int64),
    ('packingFraction', float64),
    ('sigma', float64),
    ('epsilon', float64),
    ('cutOfDistance', float64),
    ('temperature', float64),
    ('mass', float64),
    ('dimensions', int64),
    ('boxLength', float64)
]

@jitclass(spec_particle_system)
class ParticleSystem:

    def __init__(self, numberOfParticles, packingFraction, sigma, epsilon, cutOfDistance, temperature, mass, dimensions, seed=None):

        if seed is not None:
            np.random.seed(seed)

        self.numberOfParticles = numberOfParticles
        self.packingFraction = packingFraction
        self.sigma = sigma
        self.epsilon = epsilon
        self.cutOfDistance = cutOfDistance

        self.temperature = temperature
        self.mass = mass
        self.dimensions = dimensions
        self.boxLength = self.calc_box_length()


    def calc_box_length(self):
        return np.sqrt(self.numberOfParticles * np.pi * self.sigma**2 / (4.0 * self.packingFraction))
    
    def trial_position(self):
        x, y = np.random.uniform(0, self.boxLength, 2)
        return x, y
        
    def check_validity_of_position(self, x, y, particles, particleIndex):
        for i in range(particleIndex):
            dx = x - particles[i][0]
            dy = y - particles[i][1]

            dx -= self.boxLength * np.round(dx / self.boxLength)
            dy -= self.boxLength * np.round(dy / self.boxLength)

            distanceSquared = dx**2 + dy**2

            if distanceSquared < self.sigma**2:
                return False
        return True
    
    def initialize_random_particles(self):
        particles = np.zeros((self.numberOfParticles, 2))

        particleIndex = 0
        while particleIndex < self.numberOfParticles:
            x, y = self.trial_position()
            if self.check_validity_of_position(x, y, particles, particleIndex):
                particles[particleIndex, 0] = x
                particles[particleIndex, 1] = y
                particleIndex += 1
        return particles
    
    def initialize_ordered_particles(self):
        particles = np.zeros((self.numberOfParticles, 2))

        numberOfColumns = int(np.ceil(np.sqrt(self.numberOfParticles)))
        numberOfRows = int(np.ceil(self.numberOfParticles / numberOfColumns))

        dx = self.boxLength / numberOfColumns
        dy = self.boxLength / numberOfRows

        particleIndex = 0

        for row in range(numberOfRows):
            for col in range(numberOfColumns):
                if particleIndex < self.numberOfParticles:
                    x = (col + 0.5) * dx
                    y = (row + 0.5) * dy

                    if row % 2 == 1:
                        x += 0.5 * dx

                    if x >= self.boxLength:
                        x -= self.boxLength

                    particles[particleIndex, 0] = x
                    particles[particleIndex, 1] = y
                    particleIndex += 1

        return particles
    
    def initialize_particles(self):
        if self.packingFraction < 0.75:
            return self.initialize_random_particles()
        else:
            return self.initialize_ordered_particles()
    
    def initialize_velocities(self):
        velocities = np.zeros((self.numberOfParticles, 2))
        std = np.sqrt(self.temperature / self.mass)

        meanVx = 0
        meanVy = 0

        for i in range(self.numberOfParticles):
            velocities[i, 0] = np.random.normal(0.0, std)
            velocities[i, 1] = np.random.normal(0.0, std)

            meanVx += velocities[i, 0]
            meanVy += velocities[i, 1]
        
        meanVx /= self.numberOfParticles
        meanVy /= self.numberOfParticles

        velocities[:, 0] -= meanVx
        velocities[:, 1] -= meanVy

        currentTemperature = calc_system_temperature(velocities, self.mass, self.dimensions)
        scalingFactor = np.sqrt(self.temperature / currentTemperature)

        for i in range(self.numberOfParticles):
            for d in range(self.dimensions):
                velocities[i, d] *= scalingFactor
    
        return velocities
    
    
if __name__ == "__main__":
    numberOfParticles = 100
    packingFraction = 0.5
    sigma = 1.0
    epsilon = 1.0
    cutOfDistance = 2.5 * sigma
    temperature = 2.0
    mass = 1.0
    dimensions = 2

    particleSystem = ParticleSystem(numberOfParticles, packingFraction, sigma, epsilon, cutOfDistance, temperature, mass, dimensions)
    particles = particleSystem.initialize_particles()
    velocities= particleSystem.initialize_velocities()
