import numpy as np

from scripts.basic_functions import mag_vector, mkdir_event, join_npys

from classes.particles import Particles

# Trajectory of axion clump as point particle, starting from positon r0 with velocity v0x in the -x-direction
def evolve_AC(AC, NS, rprecision=1e-4, save_interval = None, conservation_check = False):
    # Instantiate axion star position as a point particles
    AC_particle = Particles([AC.center], [AC.vcenter])
    impact_param = AC_particle.impact_param()
    # Remove star if it will not reach the Roche radius
    roche = AC.roche(NS)
    AC_particle.remove_particles_far(roche, NS)
    if AC_particle.nparticles == 0:
        return None
    # Evolve until it reaches the Roche radius
    conservation_checks = AC_particle.trajectories(NS, roche, min_or_max = -1, rprecision = rprecision, save_interval = save_interval, conservation_check = conservation_check)
    AC.center, AC.vcenter = AC_particle.positions[0], AC_particle.velocities[0]

    return conservation_checks, impact_param

def draw_particles_AC(AC, nparticles, bmax, length):
    if AC.clump_type_short == 'MCNFW':
        return AC.draw_particles(nparticles, bmax, length)
    elif AC.clump_type_short == 'dAS':
        return AC.draw_particles(nparticles)

# Trajectory of nparticles from an axion clump in the field of a neutron star
def evolve_particles(AC, NS, nparticles, bmax = None, length = [-1,1], rprecision=1e-4, save_interval = None, save_file = None, conservation_check = False):   # nparticles has to be larger than one
    # Replace axion clump by collection of free particles
    particles = Particles([], [])
    all_mass_inside, total_drawn = [], 0
    while particles.nparticles < nparticles:
        positions, velocities, mass_inside = draw_particles_AC(AC, nparticles, bmax, length)
        total_drawn += len(positions)

        particles.add_particles([np.array([0.]*len(positions))], positions, velocities, np.array([[None, None, None]]*len(positions)))
        if save_interval is not None:
            particles.remove_particles_far(np.max(save_interval), NS)
        
        all_mass_inside.append(mass_inside)
    # Find trajectories
    rmax = np.max(mag_vector(particles.positions))
    return particles.trajectories(NS, rmax, rprecision = rprecision, save_interval = save_interval, save_file = save_file, conservation_check = conservation_check), np.mean(all_mass_inside), total_drawn

def evolve(NS, AC, pool, nparticles, length = [-1,1], batch_size = 1000, rprecision=1e-4, conservation_check = False):
    nbatches = int(nparticles/batch_size)
    r_in, v_in = AC.center, AC.vcenter
    if mag_vector(v_in) != -v_in[1] and AC.clump_type_short == 'MCNFW':
        return None
    impact_param = evolve_AC(AC, NS)[1][0]
    r_roche, v_roche = AC.center, AC.vcenter

    rcmax = NS.conversion_radius_max(Particles.axionmass)
    
    if AC.clump_type_short == 'MCNFW':
        bmax = AC.max_impact_param(NS)
        bmax += 3*mag_vector(AC.center)/mag_vector(AC.vcenter)*AC.circ_v([np.array([AC.radius_trunc(),0,0]) + AC.center])[0]
    else:
        bmax = None

    namedir = mkdir_event(NS, AC, r_in, v_in, impact_param, r_roche, v_roche, bmax)
    
    all_evolves = pool.starmap(evolve_particles, [(AC, NS, batch_size, bmax, length, rprecision, [NS.radius, 2*rcmax], namedir + '/' + namedir + str(i), conservation_check) for i in np.arange(nbatches)])
    conservation_checks = [evolves[0] for evolves in all_evolves]
    all_masses_in = [evolves[1] for evolves in all_evolves]
    all_total_drawn = [evolves[2] for evolves in all_evolves]
    join_npys(namedir)
    return namedir, conservation_checks, np.mean(all_masses_in), sum(all_total_drawn)
