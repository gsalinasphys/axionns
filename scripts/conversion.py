import numpy as np
import matplotlib.pyplot as plt
from scripts.basic_functions import output_dir, mag_vector
from classes.ns import NeutronStar
from classes.particles import Particles

def find_hits(single_particle, NS):
    hits = []
    single_particle = single_particle[single_particle[:, 1].argsort()]

    positions = single_particle[...,2:5]
    distances = mag_vector(positions)
    velocities = single_particle[...,5:]

    if np.min(distances) < NS.conversion_radius_max(Particles.axionmass):
        out_or_in = 1
        for i, position in enumerate(positions):
            conv_radius = NS.conversion_radius_est(position, Particles.axionmass)
            if conv_radius is not None:
                if out_or_in*mag_vector(position) < out_or_in*conv_radius:
                    out_or_in *= -1
                    hits.append([position, velocities[i]])

    return hits