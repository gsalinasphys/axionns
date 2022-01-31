import numpy as np
from scripts.basic_functions import G_N, c, mag_vector, numbers_times_vectors, sphere_point_picking, angle_between_vecs
from scipy.spatial.transform import Rotation
from scipy.optimize import root_scalar, root

# A neutron star
class NeutronStar:
    def __init__(self, mass, radius, period = 1, axis = np.array([0, 0, 1]), dipole_moment = np.array([0, 0, 1])):
        self.mass = mass    # Neutron stars' mass in solar masses
        self.radius = radius  # Radius of neutron star in km
        self.period = period    # Period of rotation in seconds
        self.axis = np.array(axis)  # Axis of rotation
        self.dipole_moment = np.array(dipole_moment)    # Magnetic dipole with magnitude in units of 10^30 Am^2

    # Find the gravitational field produced by the neutron star at some position in km/s^2
    def gravitational_field(self, positions):
        positions, distances = np.array(positions), mag_vector(positions)
        in_or_out = distances < self.radius
        return -G_N*self.mass/np.power(self.radius,3)*numbers_times_vectors(np.heaviside(in_or_out, 0.), positions) -G_N*self.mass*numbers_times_vectors(np.heaviside(np.logical_not(in_or_out), 0.)/np.power(distances,3), positions)

    # Find the gravitational potential produced by the neutron star at some position in (km/s)^2
    def gravitational_potential(self, positions):
        distances = mag_vector(positions)
        in_or_out = distances < self.radius
        return -G_N*self.mass*((np.power(self.radius, 2) - np.power(distances, 2))/(2*np.power(self.radius, 3)) + 1/self.radius)*np.heaviside(in_or_out, 0.) -G_N*self.mass*np.heaviside(np.logical_not(in_or_out), 0.)/distances

    def rotate_dipole_moment(self, time):
        angle = 2*np.pi*time/self.period
        self.dipole_moment = self.rotate_vector(self.dipole_moment, self.axis, angle)

    def magnetic_field(self, positions):   # In units of 10^18 Gauss, just the dipole contribution
        positions, distances = np.array(positions), mag_vector(positions)
        return numbers_times_vectors(3*np.dot(positions, self.dipole_moment)/np.power(distances, 5), positions) - numbers_times_vectors(1./np.power(distances, 3), np.array([self.dipole_moment]*len(positions)))

    def wplasma(self, positions):  # Plasma frequency in GHz
        return 1.5e4*np.sqrt(np.abs(np.dot(self.magnetic_field(positions), self.axis))/self.period)

    def conversion_radius_est(self, position, axionmass):    # Estimated conversion radius in some direction
        direction = np.array(position)/mag_vector(position)
        resonant_freq = axionmass*15.192669 # in GHz
        to_minimize = lambda scale: self.wplasma([scale*np.array(direction)])[0] - resonant_freq
        try:
            return root_scalar(to_minimize, bracket=[self.radius, 100*self.radius], method='brenth').root
        except ValueError:
            return None

    def conversion_surface_est(self, axionmass):
        n_avg, radii, directions = int(1e4), [], []
        for i in np.arange(n_avg):
            direction = sphere_point_picking()
            rc = self.conversion_radius_est(direction, axionmass)
            if rc is not None:
                radii.append(rc)
                directions.append(direction)
        return numbers_times_vectors(np.array(radii), np.array(directions))

    def conversion_radius_max(self, axionmass):
        old_dipole_moment = self.dipole_moment
        self.dipole_moment = mag_vector(self.dipole_moment)*self.axis
        rcmax = self.conversion_radius_est(self.axis, axionmass)
        self.dipole_moment = old_dipole_moment
        return rcmax

    def conversion_radius_exact(self, position, velocity, axionmass):
        axionmass_GHz = axionmass*15.192669
        omega2 = np.power(axionmass_GHz, 2) + np.power(axionmass_GHz*mag_vector(velocity)/c, 2) # in GHz
        theta = angle_between_vecs(self.magnetic_field([position])[0], velocity)
        resonant_freq2 = np.power(axionmass_GHz, 2)*omega2/(np.power(axionmass_GHz*np.cos(theta), 2) + omega2*np.power(np.sin(theta), 2))
        to_minimize = lambda scale: self.wplasma([scale*np.array(position)/mag_vector(position)])[0] - np.sqrt(resonant_freq2)
        try:
            return root_scalar(to_minimize, bracket=[self.radius, 100*self.radius], method='brenth').root
        except ValueError:
            return None

    @staticmethod
    def rotate_vector(vector, axis, angle):
        rotation_vector = angle*axis
        rotation = Rotation.from_rotvec(rotation_vector)
        return rotation.apply(vector) 