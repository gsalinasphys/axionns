import numpy as np
import random
from basic_functions import mag_vector

# A dilute axion star
class AxionStar:
    def __init__(self, mass, axionmass, center = [None, None, None], vcenter = [None, None, None]):
        self.mass = mass  # Axion star mass in units of 10^{-12} solar masses
        self.axionmass = axionmass
        self.center = np.array(center)
        self.vcenter = np.array(vcenter)

    # R99 is the radius that contains 99% of the axion star's mass
    def radius99(self):
        return 2.64e3/(np.power(self.axionmass, 2)*self.mass)   # Value of radius in km

    def roche(self, NS):
        return 1e4*self.radius99()*np.power(2*NS.mass/self.mass, 1/3)

    # Mass density profile for a dilute axion star, normalized so that it constains only a single axion. The maximum radius is 2*R99
    def density_profile(self):
        prob_distr = np.loadtxt("input/AS_profile_2R99_1particle.txt")
        rinterv = np.linspace(0, 2*self.radius99(), len(prob_distr))
        return rinterv, prob_distr

    # Populate the axion star with ndraw particles
    def draw_particles(self, ndraw):
        ds_from_center = np.array(random.choices(self.density_profile()[0], weights = self.density_profile()[1], k = ndraw))
        positions = self.center + np.array([self.sphere_point_picking()*ds_from_center[i] for i in np.arange(ndraw)])
        velocities = np.array([self.vcenter]*ndraw)
        return positions, velocities

    # Method to draw uniformily distributed points along the unit sphere (Marsaglia 1972)
    @staticmethod
    def sphere_point_picking(): # Maybe need to vectorize to pick many unit vectors at once
        x1, x2 = 1, 1
        xnorm = np.power(x1,2) + np.power(x2,2)
        while xnorm >= 1:
            x1, x2 = np.random.uniform(-1,1,2)
            xnorm = np.power(x1,2) + np.power(x2,2)

        nx = 2*x1*np.sqrt(1 - xnorm)
        ny = 2*x2*np.sqrt(1 - xnorm)
        nz = 1 - 2*xnorm

        return np.array([nx, ny, nz])