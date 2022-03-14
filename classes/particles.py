import numpy as np

from scripts.basic_functions import G_N, output_dir, mag_vector, numbers_times_vectors

# A collection of particles
class Particles:
    nparticles, axionmass = 0, 1 # Axion mass in units of 10^{-5} eV

    # Position, velocity and acceleration should be arrays of the same size
    def __init__(self, positions, velocities):
        self.positions = np.array(positions)    # in km
        self.velocities = np.array(velocities)  # in km/s
        self.accelerations = np.array([[None, None, None]]*len(positions))    # in km/s^2
        self.times = np.array([0.]*len(positions))  # in seconds
        
        self.nparticles += len(positions)

    # Kinetic energies in units of 10^{-5}eV*(km/s)^2
    def kin_en(self):
        return 0.5*self.axionmass*np.sum(np.power(self.velocities, 2), axis = 1)

    # Gravitational energies in units of 10^{-5}eV*(km/s)^2
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

    def verlet_step(self, NS, rprecision=1e-4):
        # Update accelerations
        self.accelerations = NS.gravitational_field(self.positions)
        # Choosing time steps carefully
        dts_both = rprecision*np.array([mag_vector(self.velocities)/mag_vector(self.accelerations), mag_vector(self.positions)/mag_vector(self.velocities)]).T
        dts = dts_both.min(axis = 1)  # Choose time step so that the percent changes in position and velocities are at most rprecision
        # Update velocities and positions
        self.verlet1(dts)
        # Update accelerations again
        self.accelerations = NS.gravitational_field(self.positions)
        # Update velocities again
        self.verlet2(dts)

        time_periodicity = 1.e8
        if np.any(self.times > time_periodicity):
            self.times -= time_periodicity

        self.times += dts
        

    # Calculate the distances of minimum approach for the particles
    def min_approach(self, NS):
        energies, ang_momenta = self.energies(NS), mag_vector(self.ang_momenta())
        rmins = -G_N*NS.mass*self.axionmass/(2*energies) + np.sign(energies)*0.5*np.sqrt(np.power(G_N*NS.mass*self.axionmass/energies, 2) + 2*self.axionmass*np.power(ang_momenta, 2)/energies)
        return rmins

    # Remove particles given their indices
    def remove_particles(self, indices):
        self.times = np.delete(self.times, indices)

        indices_torem = np.concatenate((indices*3,(indices)*3 + 1,(indices)*3 + 2))
        self.positions, self.velocities, self.accelerations = np.delete(self.positions, indices_torem), np.delete(self.velocities, indices_torem), np.delete(self.accelerations, indices_torem)
        
        self.nparticles -= len(indices)
        self.positions, self.velocities, self.accelerations = self.positions.reshape(self.nparticles, 3), self.velocities.reshape(self.nparticles, 3), self.accelerations.reshape(self.nparticles, 3)

    # Remove particles that are not gonna reach max_approach
    def remove_particles_far(self, max_approach, NS):
        indices = np.where(self.min_approach(NS) > max_approach)[0]
        self.remove_particles(indices)

    def add_particles(self, times, positions, velocities, accelerations):
        self.times = np.append(self.times, times)
        if len(self.positions) == 0:
            self.positions, self.velocities, self.accelerations = positions, velocities, accelerations
        else:
            self.positions = np.append(self.positions, positions, axis = 0)
            self.velocities = np.append(self.velocities, velocities, axis = 0)
            self.accelerations = np.append(self.accelerations, accelerations, axis = 0)
        self.nparticles += len(times[0])

    # Full trajectories bound by rmax = rlimit, change min_or_max to -1 to evolve until it reaches rmin = rlimit (only for single particle)
    def trajectories(self, NS, rlimit, min_or_max = +1, rprecision=1e-4, save_interval = None, save_file = None, conservation_check = False):
        data_list = [[] for i in np.arange(8)]

        if conservation_check:
            energy_in, ang_momenta_in, max_percent_en, max_percent_ang = self.energies(NS), self.ang_momenta(), 0, 0

        iteration = 0
        while np.any(min_or_max*mag_vector(self.positions) <= min_or_max*rlimit):
            # Update position and velocity
            self.verlet_step(NS, rprecision = rprecision)

            # Save trajectories inside save_interval to data_list
            if save_interval is not None:
                particles_inside = np.where(np.logical_and(mag_vector(self.positions) > save_interval[0], mag_vector(self.positions) < save_interval[1]))[0]
                
                if iteration%10 == 0:
                    # Save in the format [tags, times, rx, ry, rz, vx, vy, vz]
                    data_list[0].extend(particles_inside)
                    data_list[1].extend(self.times[particles_inside])
                    data_list[2].extend(self.positions.T[0][particles_inside])
                    data_list[3].extend(self.positions.T[1][particles_inside])
                    data_list[4].extend(self.positions.T[2][particles_inside])
                    data_list[5].extend(self.velocities.T[0][particles_inside])
                    data_list[6].extend(self.velocities.T[1][particles_inside])
                    data_list[7].extend(self.velocities.T[2][particles_inside])
    
            # Check energy and angular momentum consservation
            if conservation_check:
                max_percent_en_try, max_percent_ang_try = np.max(np.abs((self.energies(NS) - energy_in)/energy_in)), np.max(np.abs(mag_vector(self.ang_momenta() - ang_momenta_in)/mag_vector(ang_momenta_in)))
                if max_percent_en_try > max_percent_en:
                    max_percent_en = max_percent_en_try
                if max_percent_ang_try > max_percent_ang:
                    max_percent_ang = max_percent_ang_try
            
            iteration += 1

        data_array = np.array(data_list).T
        data_array = data_array[data_array[:, 0].argsort()]

        if save_file is not None:
            np.save(output_dir + save_file, np.array(data_array))

        if conservation_check:
            return 100*max(max_percent_ang, max_percent_en)
        else:
            return None
