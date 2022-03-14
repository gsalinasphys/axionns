import numpy as np
import random
from scipy.spatial import ConvexHull

from scripts.basic_functions import rho_eq, G_N, mag_vector

from classes.particles import Particles

# An axion minicluster
class AxionMiniclusterNFW:
    clump_type, clump_type_short = 'Axion Minicluster (NFW profile)', 'MCNFW'

    def __init__(self, mass, axionmass, delta = 1.55, concentration = 100, vdisp_type = 'Maxwell-Boltzmann', center = [None, None, None], vcenter = [None, None, None]):
        self.mass = mass  # Axion minicluster mass in units of 10^{-10} solar masses
        self.axionmass = axionmass  # In units of 10^{-5} eV
        self.delta = delta
        self.concentration = concentration
        self.vdisp_type = vdisp_type
        self.center = np.array(center)
        self.vcenter = np.array(vcenter)

    def rho_s(self):    # In units of 10^{-10}*M_Sun/km^3
        return 140*(1 + self.delta)*np.power(self.delta, 3)*rho_eq

    def rs(self):   # In km
        f_NFW = np.log(1 + self.concentration) - self.concentration/(1 + self.concentration)
        return np.power(self.mass/(4*np.pi*self.rho_s()*f_NFW), 1/3)

    def radius_trunc(self):
        return self.concentration*self.rs()

    def roche(self, NS):
        return self.radius_trunc()*np.power(2*NS.mass/(1e-10*self.mass), 1/3)

    def density_profile(self, positions):   # In units of 10^{-10}*M_Sun/km^3
        positions_from_center = positions - np.array([self.center]*len(positions))
        distances = mag_vector(positions_from_center)
        return self.rho_s()/(distances/self.rs()*np.power(1 + distances/self.rs(), 2))*np.heaviside(self.radius_trunc() - distances, 1)

    def gravitational_potential(self, positions):   # In units of (km/s)^2
        positions_from_center = positions - np.array([self.center]*len(positions))
        distances = mag_vector(positions_from_center)
        return -4e-10*np.pi*G_N*self.rho_s()*np.power(self.rs(), 3)/distances*np.log((distances + self.rs())/self.rs())

    # Enclosed mass from a given position, in units of 10^{-10} solar masses
    def encl_mass(self, positions):
        positions_from_center = positions - np.array([self.center]*len(positions))
        return 4*np.pi*self.rho_s()*np.power(self.rs(), 3)*(np.log((mag_vector(positions_from_center) + self.rs())/self.rs()) - mag_vector(positions_from_center)/(mag_vector(positions_from_center) + self.rs()))

    # Escape velocity in km/s
    def v_esc(self, positions):
        return np.sqrt(np.abs(2*self.gravitational_potential(positions)))

    def circ_v(self, positions):
        positions_from_center = positions - np.array([self.center]*len(positions))
        return 1e-5*np.sqrt(G_N*self.encl_mass(positions)/mag_vector(positions_from_center))

    def draw_vs(self, position, multiplier = 100):
        v_esc, circ_v = self.v_esc([np.array(position)])[0], self.circ_v([np.array(position)])[0]
        if self.vdisp_type == 'Maxwell-Boltzmann':
            vs_try = np.random.normal(0, circ_v, (10*multiplier,3))
            vs_dispersion = vs_try[mag_vector(vs_try) < v_esc]
            return vs_dispersion[:multiplier]

    def v_dispersion(self, positions, multiplier = 100):
        vs_dispersion = np.array([self.draw_vs(position, multiplier) for position in positions]).reshape(len(positions)*multiplier, 3)
        return vs_dispersion

    def max_impact_param(self, NS, step = 1):
        bmax, iterate_check = 0, True
        rcmax = NS.conversion_radius_max(Particles.axionmass)
        while iterate_check:
            fake_position = np.array([0, self.center[1], 0]) + np.array([bmax, 0, 0])
            fake_particle = Particles([fake_position], [self.vcenter])
            if fake_particle.min_approach(NS) < rcmax:
                bmax += step
            else:
                iterate_check = False

        return bmax

    def bmax_cylinder(self, ndraw, bmax, length = [-1,1], resolution = 1000):    # Variable 'length' is an interval contained in [-1, 1]
        rinterv = np.linspace(0, bmax, resolution)
        possible_rs = np.random.choice(rinterv, ndraw*resolution, p = rinterv/np.sum(rinterv))
        possible_thetas = np.random.uniform(0, 2*np.pi, ndraw*resolution)
        possible_ys = mag_vector(self.center) + np.random.uniform(self.radius_trunc()*length[0], self.radius_trunc()*length[1], ndraw*resolution)

        possible_xs, possible_zs = possible_rs*np.cos(possible_thetas), possible_rs*np.sin(possible_thetas)

        positions_possible = np.array([possible_xs, possible_ys, possible_zs])
        positions_possible = positions_possible.T

        ch = ConvexHull(positions_possible)
        mass_inside = np.sum(self.density_profile(positions_possible))*ch.volume/len(positions_possible)
        return positions_possible, mass_inside

    # Populate the axion minicluster with ndraw particles with a maximum impact parameter 'bmax'
    def draw_particles(self, ndraw, bmax, length = [-1,1], resolution = 1000, multiplier = 100):
        positions_possible, mass_inside = self.bmax_cylinder(ndraw, bmax, length = length, resolution = resolution)

        positions = np.array(random.choices(positions_possible, weights = self.density_profile(positions_possible), k = ndraw))

        if self.vdisp_type == 'Maxwell-Boltzmann':
            vs_dispersion = self.v_dispersion(positions, multiplier)

        velocities = np.array([self.vcenter]*len(vs_dispersion)) + vs_dispersion
        positions = np.array([[position]*multiplier for position in positions]).reshape(len(positions)*multiplier, 3)

        return positions, velocities, mass_inside