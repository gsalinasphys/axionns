import numpy as np
from scripts.basic_functions import mag_vector, mkdir_event, join_npys
from classes.particles import Particles

# Trajectory of axion clump as point particle, starting from positon r0 with velocity v0x in the -x-direction
def evolve_AC(AC, NS, rprecision=1e-4, save_interval = None, conservation_check = False):
    # Instantiate axion star position as a point particles
    AC_particle = Particles([AC.center], [AC.vcenter])
    # Remove star if it will not reach the Roche radius
    roche = AC.roche(NS)
    AC_particle.remove_particles_far(roche, NS)
    if AC_particle.nparticles == 0:
        return None
    # Evolve until it reaches the Roche radius
    conserv_check = AC_particle.trajectories(NS, roche, min_or_max = -1, rprecision = rprecision, save_interval = save_interval, conservation_check = conservation_check)
    AC.center, AC.vcenter = AC_particle.positions[0], AC_particle.velocities[0]
    return conserv_check

def draw_particles_AC(AC, nparticles, bmax):
    if AC.clump_type_short == 'MCNFW':
        bmax += 3*mag_vector(AC.center)/mag_vector(AC.vcenter)*AC.circ_v([np.array([AC.radius_trunc(),0,0]) + AC.center])[0]
        return AC.draw_particles(nparticles, bmax)
    elif AC.clump_type_short == 'dAS':
        return AC.draw_particles(nparticles)

# Trajectory of nparticles from an axion clump in the field of a neutron star
def evolve_particles(AC, NS, nparticles, bmax = None, rprecision=1e-4, save_interval = None, save_file = None, conservation_check = False):   # nparticles has to be larger than one
    # Replace axion clump by collection of free particles
    particles = Particles([], [])
    while particles.nparticles < nparticles:
        positions, velocities = draw_particles_AC(AC, nparticles, bmax)
        particles.add_particles([np.array([0.]*len(positions))], positions, velocities, np.array([[None, None, None]]*len(positions)))
        particles.remove_particles_far(np.max(save_interval), NS)
    # Find trajectories
    rmax = np.max(mag_vector(particles.positions))
    return particles.trajectories(NS, rmax, rprecision = rprecision, save_interval = save_interval, save_file = save_file, conservation_check = conservation_check)

def evolve(NS, AC, pool, nparticles, batch_size = 1000, rprecision=1e-4, conservation_check = False):
    nbatches = int(nparticles/batch_size)
    r_in, v_in = AC.center, AC.vcenter
    if mag_vector(v_in) != -v_in[1] and AC.clump_type_short == 'MCNFW':
        return None
    evolve_AC(AC, NS)
    namedir = mkdir_event(NS, AC, r_in, v_in, nparticles)
    rcmax = NS.conversion_radius_max(Particles.axionmass)
    if AC.clump_type_short == 'MCNFW':
        bmax = AC.max_impact_param(NS)
    else:
        bmax = None
    conservation_checks = pool.starmap(evolve_particles, [(AC, NS, batch_size, bmax, rprecision, [NS.radius, 2.*rcmax], namedir + '/' + namedir + str(i), conservation_check) for i in np.arange(nbatches)])
    join_npys(namedir)
    return namedir, conservation_checks