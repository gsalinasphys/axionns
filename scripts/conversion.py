import numpy as np
from scipy.interpolate import interp1d

from scripts.basic_functions import conv_factor_eV_GHz, conv_factor_G_eV2, conv_factor_km_eVinv, c, mag_vector, angle_between_vecs, crossed_zero_at

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

def find_hits(single_particle, NS, precision = 1e-3, exact = False):
    hits = []
    single_particle = single_particle[single_particle[:, 1].argsort()]

    tag = single_particle[0][0]
    times = single_particle[...,1]
    positions = single_particle[...,2:5]
    velocities = single_particle[...,5:]

    try:
        rinterp = interp1d(times, positions.T)
        vinterp = interp1d(times, velocities.T)

        to_root = lambda t: mag_vector(rinterp(t)) - NS.conversion_radius(rinterp(t), vinterp(t), t, Particles.axionmass, exact=exact)[0]
        trange = np.linspace(times[0], times[-1], int(1/precision))
        to_root_list = [to_root(time) for time in trange]
        t_hits = 0.5*(trange[crossed_zero_at(to_root_list)] + trange[crossed_zero_at(to_root_list)[0] + 1])
        
        for t_hit in t_hits:
            hits.append([tag, t_hit, rinterp(t_hit)[0], rinterp(t_hit)[1], rinterp(t_hit)[2], vinterp(t_hit)[0], vinterp(t_hit)[1], vinterp(t_hit)[2]])

        return np.array(hits)
    except:
        return hits

def find_all_hits(single_particles, NS, pool, precision = 1e-3, exact = False):
    all_hits = pool.starmap(find_hits, [(single_particle, NS, precision, exact) for single_particle in single_particles])

    all_hits = [all_hit for all_hit in all_hits if len(all_hit) > 0]
    all_hits_flat = []
    for all_hit in all_hits:
        all_hits_flat.extend(all_hit)
    
    return np.array(all_hits_flat)

def conversion_probability_est(hit, gag, NS, epsilon = 1e-8):
    axionmass_GHz = Particles.axionmass*1.e-5*conv_factor_eV_GHz

    time, position, velocity = hit[1], hit[2:5], hit[5:8]
    BNS = NS.magnetic_field([position], time)[0]
    wp = NS.wplasma([position], time)[0]

    k = axionmass_GHz*mag_vector(velocity)/c    # in GHz
    theta = angle_between_vecs(BNS, velocity)
    
    x_dir = np.cross(velocity, BNS)
    y_dir = np.cross(velocity, x_dir)
    y_hat = y_dir/mag_vector(y_dir)
    s_dir = np.cos(theta)*y_hat + np.sin(theta)*velocity/mag_vector(velocity)
    s_hat = s_dir/mag_vector(s_dir)

    new_position = position + s_hat*epsilon
    new_wp = NS.wplasma([new_position], time)[0]
    
    wp_prime = (new_wp - wp)/epsilon
    
    return conv_factor_eV_GHz*np.power(conv_factor_G_eV2, 2)*conv_factor_km_eVinv*1.e-18/2.*np.power(gag*mag_vector(BNS)*np.sin(theta), 2)*np.pi*np.power(axionmass_GHz, 5)/(2*k*np.abs(wp_prime))/np.power(np.power(k, 2) + np.power(axionmass_GHz*np.sin(theta), 2), 2)

# def conversion_probability(hit, gag, NS, epsilon = 1e-8):
#     axionmass_GHz = Particles.axionmass*1.e-5*conv_factor_eV_GHz

#     time, position, velocity = hit[1], hit[2:5], hit[5:8]
#     BNS = NS.magnetic_field([position], time)[0]
#     wp = NS.wplasma([position], time)[0]

#     gamma = 1/np.sqrt(1 - np.power(mag_vector(velocity)/c, 2))

#     k = gamma*axionmass_GHz*mag_vector(velocity)/c    # in GHz
#     omega2 = np.power(axionmass_GHz, 2) + np.power(k, 2) # in GHz^2
#     theta = angle_between_vecs(BNS, velocity)
#     wp_bar2 = np.power(axionmass_GHz, 2)*omega2/(np.power(axionmass_GHz*np.cos(theta), 2) + omega2*np.power(np.sin(theta), 2))  # in GHz^2
    
#     x_dir = np.cross(velocity, BNS)
#     y_dir = np.cross(velocity, x_dir)
#     y_hat = y_dir/mag_vector(y_dir)

#     s_dir = wp_bar2*np.cos(theta)*np.sin(theta)/(omega2 - wp_bar2*np.power(np.cos(theta), 2))*y_hat + velocity/mag_vector(velocity)
#     s_hat = s_dir/mag_vector(s_dir)

#     new_position = position + s_hat*epsilon
#     new_BNS = NS.magnetic_field([new_position], time)[0]
#     new_wp = NS.wplasma([new_position], time)[0]
    
#     new_theta = angle_between_vecs(new_BNS, velocity)
    
#     wp_bar_prime = (new_wp - wp)/epsilon
#     theta_prime = (new_theta - theta)/epsilon
    
#     Ey_over_a2 = gag*BNS*omega2*np.sqrt(np.pi/(2.*k*(wp*wp_bar_prime + (omega2 - np.power(wp, 2))/(omega2*np.tan(theta))*np.power(wp, 2)*theta_prime)))
#     U_over_a2 = Ey_over_a2*((2*np.power(omega2, 2) - omega2*np.power(wp, 2) + (np.power(wp, 4) - 3*omega2*np.power(wp, 2))*np.cos(2*theta) + np.power(wp, 4))/(8*np.power(omega2 - np.power(wp*np.cos(theta), 2), 2) + np.power(k, 2)/(4*omega2))
#     return 2*U_over_a2/omega2

def conversion_probabilities_est(all_hits, gag, NS, pool, epsilon = 1e-8):
    return pool.starmap(conversion_probability_est, [(hit, gag, NS, epsilon) for hit in all_hits])
