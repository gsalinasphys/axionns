import numpy as np
from basic_functions import mag_vector, G_N, numbers_times_vectors

# A neutron star
class NeutronStar:
    def __init__(self, mass, radius):
        self.mass = mass    # Neutron stars' mass in solar masses
        self.radius = radius  # Radius of neutron star in km

    # Find the gravitational field produced by the neutron star at some position
    def gravitational_field(self, positions):
        distances = mag_vector(positions)
        in_or_out = distances < self.radius
        return -G_N*self.mass/np.power(self.radius,3)*numbers_times_vectors(np.heaviside(in_or_out, 0.), positions) -G_N*self.mass*numbers_times_vectors(np.heaviside(np.logical_not(in_or_out), 0.)/np.power(distances,3), positions)

    # Find the gravitational potential produced by the neutron star at some position
    def gravitational_potential(self, positions):
        distances = mag_vector(positions)
        in_or_out = distances < self.radius
        return -G_N*self.mass*((np.power(self.radius, 2) - np.power(distances, 2))/(2*np.power(self.radius, 3)) + 1/self.radius)*np.heaviside(in_or_out, 0.) -G_N*self.mass*np.heaviside(np.logical_not(in_or_out), 0.)/distances

    def conversion_radius(self):
        return 100