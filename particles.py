import numpy as np
import pandas as pd
from basic_functions import mag_vector, numbers_times_vectors, G_N

# A collection of particles
class Particles:
    nparticles, axionmass = 0, 2.6 # Axion mass in units of 10^{-5} eV

    # Position, velocity and acceleration should be arrays of the same size
    def __init__(self, positions, velocities):
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.accelerations = np.array([None, None, None]*len(positions))
        
        self.nparticles += len(positions)

    # Kinetic energies
    def kin_en(self):
        return 0.5*self.axionmass*np.sum(np.power(self.velocities, 2), axis = 1)

    # Gravitational energies
    def grav_en(self, NS):
        return self.axionmass*NS.gravitational_potential(self.positions)

    # Total energies
    def energies(self, NS):
        return self.kin_en() + self.grav_en(NS)

    # Angular momenta
    def ang_momenta(self):
        return np.cross(self.positions, self.velocities)

    # Implementation of Verlet's method to update position and velocity (for reference, see Velocity Verlet in https://en.m.wikipedia.org/wiki/Verlet_integration)
    def verlet1(self, dts):
        self.velocities += 0.5*numbers_times_vectors(dts, self.accelerations)
        self.positions += numbers_times_vectors(dts, self.velocities)

    def verlet2(self, dts):
        self.velocities += 0.5*numbers_times_vectors(dts, self.accelerations)

    def verlet_step(self, NS, rprecision):
        # Update accelerations
        self.accelerations = NS.gravitational_field(self.positions)
        # Choosing time steps carefully
        # dts_both = rprecision*np.array([mag_vector(self.velocities)/mag_vector(self.accelerations), mag_vector(self.positions)/mag_vector(self.velocities)]).T
        # dts = dts_both.min(axis = 1)  # Choose time step so that the percent changes in position and velocities are at most rprecision
        dts = rprecision*mag_vector(self.positions)/mag_vector(self.velocities)
        # Update velocities and positions
        self.verlet1(dts)
        # Update accelerations again
        self.accelerations = NS.gravitational_field(self.positions)
        # Update velocities again
        self.verlet2(dts)

    # Calculate the distances of minimum approach for the particles
    def min_approach(self, NS):
        energies, ang_momenta = self.energies(NS), mag_vector(self.ang_momenta())
        rmins = -G_N*NS.mass*self.axionmass/(2*energies) + np.sign(energies)*0.5*np.sqrt(np.power(G_N*NS.mass*self.axionmass/energies, 2) + 2*self.axionmass*np.power(ang_momenta, 2)/energies)
        return rmins

    # Remove particles given their indices
    def remove_particles(self, indices):
        indices_torem = np.concatenate((indices*3,(indices)*3 + 1,(indices)*3 + 2))
        self.positions, self.velocities, self.accelerations = np.delete(self.positions, indices_torem), np.delete(self.velocities, indices_torem), np.delete(self.accelerations, indices_torem)
        
        self.nparticles -= len(indices)
        self.positions, self.velocities, self.accelerations = self.positions.reshape(self.nparticles, 3), self.velocities.reshape(self.nparticles, 3), self.accelerations.reshape(self.nparticles, 3)

    def remove_particles_far(self, max_approach, NS):
        indices = np.where(self.min_approach(NS) > max_approach)[0]
        self.remove_particles(indices)

    # Full trajectories inside a radius rmax = rlimit, change min_or_max to -1 to evolve until it reaches rmin = rlimit (only for single particle)
    def trajectories(self, NS, rlimit, min_or_max = +1, rprecision=1e-3, conservation_check = False):
        ppositions, pvelocities = [], []
        energy_in, ang_momenta_in, max_percent_en, max_percent_ang = self.energies(NS), self.ang_momenta(), 0, 0
        while np.any(min_or_max*mag_vector(self.positions) < min_or_max*rlimit):
            # Save positions and velocities
            ppositions.append(np.array(self.positions))
            pvelocities.append(np.array(self.velocities))

            # Update position and velocity
            self.verlet_step(NS, rprecision)

            # Check energy and angular momentum consservation
            if conservation_check:
                max_percent_en_try, max_percent_ang_try = np.max(np.abs((self.energies(NS) - energy_in)/energy_in)), np.max(np.abs(mag_vector(self.ang_momenta() - ang_momenta_in)/mag_vector(ang_momenta_in)))
                if max_percent_en_try > max_percent_en:
                    max_percent_en = max_percent_en_try
                if max_percent_ang_try > max_percent_ang:
                    max_percent_ang = max_percent_ang_try

        ps_and_vs = np.array([ppositions, pvelocities])
        return np.transpose(ps_and_vs, axes = (2, 0, 1, 3)), max_percent_en*100, max_percent_ang*100