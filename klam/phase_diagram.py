from initialization import ParticleSystem
from force_calculation import calc_force_potential
from particle_dynamics import velocity_verlet_step
from radial_distribution import calc_radial_distribution
import numpy as np
import matplotlib.pyplot as plt


simulationCases = [
    {"label": "gas", "temperature": 5.0, "packingFraction": 0.05},
    {"label": "liquid", "temperature": 0.8, "packingFraction": 0.4},
    {"label": "solid", "temperature": 0.05, "packingFraction": 0.9},
]

numberOfParticles = 250
sigma = 1.0
epsilon = 1.0
cutOfDistance = 2.0 * sigma
mass = 1.0

dt = 0.005
numberOfSteps = 10000
dimensions = 2
numberOfRValues = 300
delta = 0.01
thermostatInterval = 10

equilibrationSteps = numberOfSteps // 2
sampleInterval = 10

fig, axs = plt.subplots(len(simulationCases), 2, figsize=(12, 12))

for row, case in enumerate(simulationCases):

    label = case["label"]
    targetTemperature = case["temperature"]
    packingFraction = case["packingFraction"]

    system = ParticleSystem(numberOfParticles, packingFraction, sigma, epsilon, cutOfDistance, targetTemperature, mass, dimensions, seed=42)

    particles = system.initialize_particles()
    velocities = system.initialize_velocities()

    forces, potential = calc_force_potential(particles, system.boxLength, sigma, epsilon, cutOfDistance)

    gSum = np.zeros(numberOfRValues)
    numberOfSamples = 0

    for step in range(1, numberOfSteps + 1):

        particles, velocities, forces, potential = velocity_verlet_step(particles, velocities, forces, mass, dt, system.boxLength, sigma, epsilon, cutOfDistance, True, targetTemperature, dimensions, delta, thermostatInterval, step)

        if step > equilibrationSteps and step % sampleInterval == 0:
            rValues, g = calc_radial_distribution(particles, numberOfRValues, system.boxLength)
            gSum += g
            numberOfSamples += 1

    if numberOfSamples > 0:
        gAverage = gSum / numberOfSamples
    else:
        rValues, gAverage = calc_radial_distribution(particles, numberOfRValues, system.boxLength) 


    axs[row, 0].scatter(particles[:, 0], particles[:, 1], s=15)

    axs[row, 0].set_xlim(0, system.boxLength)
    axs[row, 0].set_ylim(0, system.boxLength)
    axs[row, 0].set_aspect("equal")
    axs[row, 0].set_title(f"{label}\n" f"T={targetTemperature}, packing fraction={packingFraction}")
    axs[row, 0].set_xlabel("x")
    axs[row, 0].set_ylabel("y")
    axs[row, 0].grid()

    axs[row, 1].plot(rValues, gAverage)
    axs[row, 1].axhline(1.0, linestyle="--")
    axs[row, 1].set_title( f"{label}\n" f"T={targetTemperature}, packing fraction={packingFraction}")
    axs[row, 1].set_xlabel("r")
    axs[row, 1].set_ylabel(r"$g(r)$")
    axs[row, 1].set_xlim(0, 10)
    axs[row, 1].grid()

plt.tight_layout()
plt.show()