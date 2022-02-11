import numpy as np
from scripts.basic_functions import mag_vector
from classes.particles import Particles

def divide_into_singles(part_trajs):
    indices_np = [0]
    old_tag = 0
    for i, part_traj in enumerate(part_trajs):
        tag = part_traj[0]
        if tag > old_tag:
            indices_np.append(i)
            old_tag += 1

    single_particles = []
    for i in np.arange(1, len(indices_np)):
        single_particles.append(part_trajs[indices_np[i - 1]: indices_np[i]])

    return single_particles


def find_hits(single_particle, NS):
    hits = []
    single_particle = single_particle[single_particle[:, 1].argsort()]

    times = single_particle[...,1]
    positions = single_particle[...,2:5]
    distances = mag_vector(positions)
    velocities = single_particle[...,5:]

    if np.min(distances) < NS.conversion_radius_max(Particles.axionmass):
        out_or_in = 1
        for i, position in enumerate(positions):
            conv_radius = NS.conversion_radius_est(position, times[i], Particles.axionmass)
            if conv_radius is not None:
                if out_or_in*mag_vector(position) < out_or_in*conv_radius:
                    out_or_in *= -1
                    hits.append([single_particle[...,0][0], times[i], position[0], position[1], position[2], velocities[i][0], velocities[i][1], velocities[i][2]])

    return np.array(hits)

def find_all_hits(single_particles, NS, pool):
    all_hits = pool.starmap(find_hits, [(single_particle, NS) for single_particle in single_particles])

    all_hits = [all_hit for all_hit in all_hits if len(all_hit) > 0]
    all_hits_flat = []
    for all_hit in all_hits:
        all_hits_flat.extend(all_hit)
    
    return all_hits_flat