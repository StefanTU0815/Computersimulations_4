from initialization import ParticleSystem
from force_calculation import calc_force_potential
from particle_dynamics import velocity_verlet_step_unwrapped

import numpy as np
import numba as nb
import matplotlib.pyplot as plt


@nb.njit
def calc_mean_squared_displacement(unwrappedPositions, initialUnwrappedPositions):
    displacements = unwrappedPositions - initialUnwrappedPositions
    squaredDisplacements = np.sum(displacements**2, axis=1)
    return np.mean(squaredDisplacements)


def calc_diffusion_constant_MSD(times, msd, dimensions):
    startIndex = len(times) // 2
    slope, intercept = np.polyfit(times[startIndex:], msd[startIndex:], 1)
    fitLine = slope * times + intercept
    diffusionConstant = slope / (2.0 * dimensions)
    return diffusionConstant, fitLine, startIndex


def calc_velocity_autocorrelation_fft(velocitiesOverTime):
    numberOfSteps = velocitiesOverTime.shape[0]
    numberOfParticles = velocitiesOverTime.shape[1]
    dimensions = velocitiesOverTime.shape[2]

    velocities = velocitiesOverTime.copy()

    for step in range(numberOfSteps):
        meanVelocity = np.mean(velocities[step], axis=0)
        velocities[step] -= meanVelocity

    flattenedVelocities = velocities.reshape(numberOfSteps, numberOfParticles * dimensions)

    paddedLength = 2 * numberOfSteps
    velocitiesFourier = np.fft.fft(flattenedVelocities, n=paddedLength, axis=0)
    powerSpectrum = velocitiesFourier * np.conjugate(velocitiesFourier)

    autocorrelation = np.fft.ifft(powerSpectrum, axis=0).real[:numberOfSteps]

    normalization = np.arange(numberOfSteps, 0, -1).reshape(numberOfSteps, 1)
    autocorrelation /= normalization

    velocityAutocorrelation = np.sum(autocorrelation, axis=1) / numberOfParticles

    return velocityAutocorrelation


@nb.njit
def calc_diffusion_constant_VACF(times, velocityAutocorrelation, dimensions):
    diffusionConstant = np.zeros(len(times))

    for step in range(1, len(times)):
        dt = times[step] - times[step - 1]
        diffusionConstant[step] = diffusionConstant[step - 1] + 0.5 * (velocityAutocorrelation[step] + velocityAutocorrelation[step - 1]) * dt / dimensions

    return diffusionConstant


numberOfParticles = 250
packingFraction = 0.4
targetTemperature = 1.5

sigma = 1.0
epsilon = 1.0
cutOfDistance = 2.0 * sigma
mass = 1.0
dimensions = 2

dt = 0.001
equilibrationSteps = 5000
numberOfSteps = 20000

delta = 0.01
thermostatInterval = 10

system = ParticleSystem(numberOfParticles, packingFraction, sigma, epsilon, cutOfDistance, targetTemperature, mass, dimensions, seed=1)
particles = system.initialize_particles()
unwrappedParticles = particles.copy()
velocities = system.initialize_velocities()
forces, potential = calc_force_potential(particles, system.boxLength, sigma, epsilon, cutOfDistance)

for step in range(1, equilibrationSteps + 1):
    particles, unwrappedParticles, velocities, forces, potential = velocity_verlet_step_unwrapped(particles, unwrappedParticles, velocities, forces, mass, dt, system.boxLength, sigma, epsilon, cutOfDistance, True, targetTemperature, dimensions, delta, thermostatInterval, step)

initialUnwrappedPositions = unwrappedParticles.copy()

times = np.zeros(numberOfSteps + 1)
msd = np.zeros(numberOfSteps + 1)
velocitiesOverTime = np.zeros((numberOfSteps + 1, numberOfParticles, dimensions))

velocitiesOverTime[0] = velocities

for step in range(1, numberOfSteps + 1):
    particles, unwrappedParticles, velocities, forces, potential = velocity_verlet_step_unwrapped(particles, unwrappedParticles, velocities, forces, mass, dt, system.boxLength, sigma, epsilon, cutOfDistance, False, targetTemperature, dimensions, delta, thermostatInterval, step)
    times[step] = step * dt
    msd[step] = calc_mean_squared_displacement(unwrappedParticles, initialUnwrappedPositions)
    velocitiesOverTime[step] = velocities

diffusionConstantMSD, fitLine, startIndex = calc_diffusion_constant_MSD(times, msd, dimensions)

velocityAutocorrelation = calc_velocity_autocorrelation_fft(velocitiesOverTime)
diffusionConstantVACF = calc_diffusion_constant_VACF(times, velocityAutocorrelation, dimensions)

print(f"D from MSD = {diffusionConstantMSD:.6f}")
print(f"D from VACF = {diffusionConstantVACF[-1]:.6f}")

fig, axs = plt.subplots(3, 1, figsize=(9, 11))

axs[0].plot(times, msd, label=r"$\langle \Delta r(t)^2 \rangle$")
axs[0].plot(times[startIndex:], fitLine[startIndex:], "--", label=f"linear fit, D = {diffusionConstantMSD:.4f}")
axs[0].set_xlabel("time")
axs[0].set_ylabel(r"$\langle \Delta r(t)^2 \rangle$")
axs[0].set_title("Mean squared displacement")
axs[0].legend()
axs[0].grid()

axs[1].plot(times, velocityAutocorrelation, label=r"$C_v(t)$")
axs[1].axhline(0.0, linestyle="--")
axs[1].set_xlabel("time")
axs[1].set_ylabel(r"$\langle \vec v(0)\cdot \vec v(t)\rangle$")
axs[1].set_title("Velocity autocorrelation function from FFT")
axs[1].legend()
axs[1].grid()

axs[2].plot(times, diffusionConstantVACF, label=f"D from VACF = {diffusionConstantVACF[-1]:.4f}")
axs[2].axhline(diffusionConstantMSD, linestyle="--", label=f"D from MSD = {diffusionConstantMSD:.4f}")
axs[2].set_xlabel("time")
axs[2].set_ylabel("D")
axs[2].set_title("Diffusion constant")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()