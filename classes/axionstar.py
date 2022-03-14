import numpy as np
import random

from scripts.basic_functions import G_N, sphere_point_picking, mag_vector

# A dilute axion star
class AxionStar:
    clump_type, clump_type_short = 'Dilute axion star', 'dAS'
    prob_distr = np.load("input/AS_profile_2R99.npy")

    def __init__(self, mass, axionmass, vdisp_type = None, center = [None, None, None], vcenter = [None, None, None]):
        self.mass = mass  # Axion star mass in units of 10^{-12} solar masses
        self.axionmass = axionmass  # In units of 10^{-5} eV
        self.vdisp_type = vdisp_type
        self.center = np.array(center)
        self.vcenter = np.array(vcenter)

    # R99 is the radius that contains 99% of the axion star's mass
    def radius99(self):
        return 2.64e3/(np.power(self.axionmass, 2)*self.mass)   # Value of radius in km

    def roche(self, NS):
        return 1e4*self.radius99()*np.power(2*NS.mass/self.mass, 1/3)

    # Mass density profile for a dilute axion star, normalized to the total mass of the star. The maximum radius is 2*R99
    def density_profile(self):
        rinterv = np.linspace(0, 2*self.radius99(), len(self.prob_distr))
        norm_prob_distr = self.mass/(0.22902745717696624*np.power(self.radius99(), 3))*self.prob_distr
        return rinterv, norm_prob_distr

    # Enclosed mass from a given position, in units of 10^{-12} solar masses
    def encl_mass(self, positions):
        positions_from_center = positions - np.array([self.center]*len(positions))
        indices_r99 = mag_vector(positions_from_center)/(2*self.radius99())*len(self.density_profile()[0])
        indices_r99 = indices_r99.astype(int)
        return 4*np.pi*np.array([np.trapz(self.density_profile()[1][:index_r99]*np.power(self.density_profile()[0][:index_r99], 2), self.density_profile()[0][:index_r99]) for index_r99 in indices_r99])

    def gravitational_potential(self, positions):   # In units of (km/s)^2
        positions_from_center = positions - np.array([self.center]*len(positions))
        indices_r99 = mag_vector(positions_from_center)/(2*self.radius99())*len(self.density_profile()[0])
        indices_r99 = indices_r99.astype(int)
        first_term = -G_N*self.encl_mass(positions)/mag_vector(positions_from_center)
        second_term = -4*np.pi*G_N*np.array([np.trapz(self.density_profile()[1][index_r99:]*self.density_profile()[0][index_r99:], self.density_profile()[0][index_r99:]) for index_r99 in indices_r99])
        return 1e-12*(first_term + second_term)
       
    # Escape velocity in km/s
    def v_esc(self, positions):
        return np.sqrt(np.abs(2*self.gravitational_potential(positions)))

    # Populate the axion star with ndraw particles
    def draw_particles(self, ndraw):
        ds_from_center = np.array(random.choices(self.density_profile()[0], weights = np.power(self.density_profile()[0], 2)*self.density_profile()[1], k = ndraw))
        deltas = np.array([sphere_point_picking()*ds_from_center[i] for i in np.arange(ndraw)])
        positions = self.center + deltas
        vs_esc = self.v_esc(positions)
        velocities = np.array([self.vcenter]*ndraw)

        if self.vdisp_type == 'Flat':
            vmags = np.random.uniform(0, 1, ndraw)*vs_esc
            velocities += np.array([sphere_point_picking()*vmags[i] for i in np.arange(ndraw)])
        elif self.vdisp_type == 'Escape velocity':
            vmags = vs_esc*np.ones(ndraw)
            velocities += np.array([sphere_point_picking()*vmags[i] for i in np.arange(ndraw)])
        
        return positions, velocities