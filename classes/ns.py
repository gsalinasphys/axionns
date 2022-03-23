import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import root_scalar

from scripts.basic_functions import G_N, c, conv_factor_eV_GHz, mag_vector, numbers_times_vectors, sphere_point_picking, angle_between_vecs

# A neutron star
class NeutronStar:
    def __init__(self, mass, radius, period = 1, axis = np.array([0, 0, 1.]), Bsurface = 1., misalign = 0., Psi0 = 0.):
        self.mass = mass    # Neutron stars' mass in solar masses
        self.radius = radius  # Radius of neutron star in km
        self.period = period    # Period of rotation in seconds
        self.axis = np.array(axis)  # Axis of rotation
        self.Bsurface = Bsurface    # Magnetic field at the surface in units of 10^14 G
        self.misalign = misalign    # Misalignement angle
        self.Psi0 = Psi0  # Initial azimuthal angle

    def dipole_moment(self, time):    # Magnetic dipole with magnitude in units of (10^14 G)*km^3
        Psi = self.Psi0 + 2*np.pi/self.period*time
        return 0.5*self.Bsurface*np.power(self.radius, 3)*np.array([np.sin(self.misalign)*np.sin(Psi), np.sin(self.misalign)*np.cos(Psi), np.cos(self.misalign)])

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

    def magnetic_field(self, positions, time):   # In units of 10^14 Gauss, just the dipole contribution
        positions, distances = np.array(positions), mag_vector(positions)
        return numbers_times_vectors(3*np.dot(positions, self.dipole_moment(time))/np.power(distances, 5), positions) - numbers_times_vectors(1./np.power(distances, 3), np.array([self.dipole_moment(time)]*len(positions)))

    def wplasma(self, positions, time):  # Plasma frequency in GHz
        return 1.5e2*np.sqrt(np.abs(np.dot(self.magnetic_field(positions, time), self.axis))/self.period)

    def conversion_radius_est(self, position, time, axionmass):    # Estimated conversion radius in some direction
        direction = np.array(position)/mag_vector(position)
        axionmass_GHz = axionmass*1.e-5*conv_factor_eV_GHz # in GHz
        to_minimize = lambda scale: self.wplasma([scale*np.array(direction)], time)[0] - axionmass_GHz
        try:
            rc = root_scalar(to_minimize, bracket=[self.radius, 100*self.radius], method='brenth').root
            percent_diff = -(self.wplasma([position], time)[0] - axionmass_GHz)/axionmass_GHz
            if rc > self.radius and rc < 100*self.radius:
                return rc, percent_diff
        except ValueError:
            return None

    def conversion_surface_est(self, time, axionmass):
        n_avg, radii, directions = int(1e4), [], []
        for i in np.arange(n_avg):
            direction = sphere_point_picking()
            rc = self.conversion_radius_est(direction, time, axionmass)
            if rc is not None:
                radii.append(rc[0])
                directions.append(direction)
        return numbers_times_vectors(np.array(radii), np.array(directions))

    def conversion_radius_max(self, axionmass):
        old_misalign = self.misalign
        self.misalign = 0.
        rcmax = self.conversion_radius_est(self.axis, 0., axionmass)
        self.misalign = old_misalign
        return rcmax[0]

    def conversion_radius_exact(self, position, velocity, time, axionmass):
        axionmass_GHz = axionmass*1.e-5*conv_factor_eV_GHz
        omega2 = np.power(axionmass_GHz, 2) + np.power(axionmass_GHz*mag_vector(velocity)/c, 2) # in GHz
        theta = angle_between_vecs(self.magnetic_field([position], time)[0], velocity)
        resonant_freq2 = np.power(axionmass_GHz, 2)*omega2/(np.power(axionmass_GHz*np.cos(theta), 2) + omega2*np.power(np.sin(theta), 2))
        to_minimize = lambda scale: self.wplasma([scale*np.array(position)/mag_vector(position)], time)[0] - np.sqrt(resonant_freq2)
        try:
            rc = root_scalar(to_minimize, bracket=[self.radius, 100*self.radius], method='brenth').root
            percent_diff = -(self.wplasma([position], time)[0] - np.sqrt(resonant_freq2))/np.sqrt(resonant_freq2)
            return rc, percent_diff
        except ValueError:
            return None

    def conversion_radius(self, position, velocity, time, axionmass, exact = False):
        if exact:
            return self.conversion_radius_exact(position, velocity, time, axionmass)
        elif not exact:
            return self.conversion_radius_est(position, time, axionmass)
    
    @staticmethod
    def rotate_vector(vector, axis, angle):
        rotation_vector = angle*axis
        rotation = Rotation.from_rotvec(rotation_vector)
        return rotation.apply(vector) 