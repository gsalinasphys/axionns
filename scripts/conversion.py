import numpy as np
from scripts.basic_functions import mag_vector, conv_factor_eV_GHz, conv_factor_G_eV2, conv_factor_km_eVinv, c, angle_between_vecs
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

def conversion_probability(hit, gag, NS, epsilon = 1e-8):
    axionmass_GHz = Particles.axionmass*1.e-5*conv_factor_eV_GHz

    time, position, velocity = hit[1], hit[2:5], hit[5:8]
    BNS = NS.magnetic_field([position], time)[0]
    wp = NS.wplasma([position], time)[0]

    gamma = 1/np.sqrt(1 - np.power(mag_vector(velocity)/c, 2))

    k = gamma*axionmass_GHz*mag_vector(velocity)/c    # in GHz
    omega2 = np.power(axionmass_GHz, 2) + np.power(k, 2) # in GHz^2
    theta = angle_between_vecs(BNS, velocity)
    wp_bar2 = np.power(axionmass_GHz, 2)*omega2/(np.power(axionmass_GHz*np.cos(theta), 2) + omega2*np.power(np.sin(theta), 2))  # in GHz^2
    
    x_dir = np.cross(velocity, BNS)
    y_dir = np.cross(velocity, x_dir)
    y_hat = y_dir/mag_vector(y_dir)

    s_dir = wp_bar2*np.cos(theta)*np.sin(theta)/(omega2 - wp_bar2*np.power(np.cos(theta), 2))*y_hat + velocity/mag_vector(velocity)
    s_hat = s_dir/mag_vector(s_dir)

    new_position = position + s_hat*epsilon
    new_wp = NS.wplasma([new_position], time)[0]
    
    wp_bar_prime = (new_wp - wp)/epsilon
    
    return conv_factor_eV_GHz*np.power(conv_factor_G_eV2, 2)*conv_factor_km_eVinv*1.e-18/2.*np.power(gag*mag_vector(BNS)*np.sin(theta), 2)*np.pi*np.power(axionmass_GHz, 5)/(2*k*np.abs(wp_bar_prime))/np.power(np.power(k, 2) + np.power(axionmass_GHz*np.sin(theta), 2), 2)

def conversion_probabilities(all_hits, gag, NS, pool, epsilon = 1e-8):
    return pool.starmap(conversion_probability, [(hit, gag, NS, epsilon) for hit in all_hits])
