import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import shutil
import time
from mpl_toolkits.mplot3d import Axes3D

from classes.ns import NeutronStar
from classes.particles import Particles

from scripts.basic_functions import output_dir, local_run
from scripts.orbits import evolve
from scripts.conversion import conversion_probabilities_est, divide_into_singles, find_all_hits
from scripts.event import choose_clump, axions_per_traj, add_to_readme, add_to_readme_probs

plt.rcParams['figure.figsize'] = [24, 16]

start_time = time.time()
draw_plots = False

############################## Define parameters here ###################################
M_NS, R_NS = 1, 10

MC_or_AS = 'AS'
Mass = 1.   # In units of 10^{-10} M_solar or 10^{-12} M_solar
vy_in, b = -200., 0.2   # b in units of MC.radius_trunc() or AS.radius99()
delta, concentration = 1.55, 100

length, nparticles, batch_size, conservation_check, rprecision = np.array([-1,1])/300, int(30000), int(2000), False, 1e-3

############################ Compute trajectories ###########################################
chosen_clump = choose_clump(Mass, vy_in, b, MC_or_AS = MC_or_AS, delta = delta, concentration = concentration)
NS = NeutronStar(M_NS, R_NS)

if __name__ == '__main__':
    mp.freeze_support()
    ncores = mp.cpu_count() - local_run
    pool = mp.Pool(ncores)

    event, conservation_check_result, mass_in, total_drawn = evolve(NS, chosen_clump, pool, nparticles, length = length, batch_size = batch_size, conservation_check = conservation_check, rprecision=rprecision)

part_trajs = np.load(output_dir + event + '/' + event + '.npy')
nparticles = int(part_trajs[-1][0]) + 1
naxions_per_traj = axions_per_traj(chosen_clump, total_drawn, mass_in)
add_to_readme(event, part_trajs, chosen_clump, naxions_per_traj, conservation_check, conservation_check_result, length = length)

############################ Check neighboor cylinders ###########################################
if draw_plots:
    if MC_or_AS == 'MC':
        ncheck = ncores*int(1e1)
        chosen_clump = choose_clump(Mass, vy_in, b, MC_or_AS = MC_or_AS, delta = delta, concentration = concentration)
        event_up = evolve(NS, chosen_clump, pool, ncheck, length = length + (length[1] - length[0]), batch_size = int(1e1), conservation_check = conservation_check, rprecision=rprecision)
        chosen_clump = choose_clump(Mass, vy_in, b, MC_or_AS = MC_or_AS, delta = delta, concentration = concentration)
        event_down = evolve(NS, chosen_clump, pool, ncheck, length = length - (length[1] - length[0]), batch_size = int(1e1), conservation_check = conservation_check, rprecision=rprecision)

        part_trajs_up, part_trajs_down = np.load(output_dir + event_up[0] + '/' + event_up[0] + '.npy'), np.load(output_dir + event_down[0] + '/' + event_down[0] + '.npy')
        single_particles_up, single_particles_down = divide_into_singles(part_trajs_up), divide_into_singles(part_trajs_down)
        
        shutil.rmtree(output_dir + event_up[0])
        shutil.rmtree(output_dir + event_down[0])
        
        np.save(output_dir + event + '/' + event + '_up.npy', part_trajs_up)
        np.save(output_dir + event + '/' + event + '_down.npy', part_trajs_down)
        
############################ Plot some trajectories ###########################################
if draw_plots:
    ax = plt.gca()
    nsamples = np.min([int(1e2), nparticles])
    part_trajs_cut = part_trajs[:int(nsamples*float(len(part_trajs))/nparticles)]
    for i in np.arange(int(part_trajs_cut[-1][0])):
        traj_chosen = np.array([[part_traj[3], part_traj[2]] for part_traj in part_trajs_cut if part_traj[0] == i]).T
        ax.scatter(traj_chosen[0], traj_chosen[1], s = 1)

    circle1 = plt.Circle((0, 0), NS.radius, facecolor='purple', alpha = 0.5)
    ax.add_patch(circle1)
    ax.set_aspect('equal')

    plt.xlabel('y')
    plt.ylabel('x')
    plt.savefig(output_dir + event + '/' + event + '.png')
    plt.close()

############################ Conversion in the NS magnetosphere #####################################
single_particles = divide_into_singles(part_trajs)

if __name__ == '__main__':
    mp.freeze_support()
    ncores = mp.cpu_count() - local_run
    pool = mp.Pool(ncores)
    
    all_hits = find_all_hits(single_particles, NS, pool)
    
    if draw_plots:
        if MC_or_AS == 'MC':
            all_hits_up = find_all_hits(single_particles_up, NS, pool)
            all_hits_down = find_all_hits(single_particles_down, NS, pool)

np.save(output_dir + event + '/' + event + '_conversion.npy', all_hits)

############################ Histogram of hits with time ###########################################
all_ts = np.array(all_hits).T[1]
min_t = np.min(all_ts)
all_ts -= min_t
all_ts /= 3600.

nbins = 50
hist_ts = np.histogram(all_ts, bins = np.linspace(0, np.max(all_ts), 10*nbins))
max_hist = np.max(hist_ts[0])
indices_hist = np.where(hist_ts[0] > 0.9*max_hist)
index_max_t, index_min_t = indices_hist[0][-1] + 1, indices_hist[0][0] - 1
max_t, min_t = hist_ts[1][index_max_t], hist_ts[1][index_min_t]#
indices = np.where(np.logical_and(all_ts > min_t, all_ts < max_t))
chosen_hits = all_hits[indices]

if draw_plots:
    plt.hist(all_ts, bins = np.linspace(0, np.max(all_ts), nbins));

    if MC_or_AS == 'MC':
        all_ts_up, all_ts_down = np.array(all_hits_up).T[1], np.array(all_hits_down).T[1]
        all_ts_up -= min_t
        all_ts_up /= 3600.
        plt.hist(all_ts_up, bins=np.linspace(np.min(all_ts_up), np.max(all_ts_up), nbins));
        all_ts_down -= min_t
        all_ts_down /= 3600.
        plt.hist(all_ts_down, bins=np.linspace(np.min(all_ts_down), np.max(all_ts_down), nbins));

    plt.xlabel('Hour')
    plt.ylabel('Conversion events')
    plt.yscale('log')

    plt.savefig(output_dir + event + '/' + event + '_hist.png')
    plt.close()

############################ Plot conversion surface ###########################################

if draw_plots:
    X, Y, Z = np.array(chosen_hits).T[2:5]
    ax = plt.gca(projection='3d')
    ax.scatter3D(X, Y, Z);

    Xsurf, Ysurf, Zsurf = NS.conversion_surface_est(0., Particles.axionmass).T
    ax.scatter3D(Xsurf, Ysurf, Zsurf, c = 'purple', alpha = 0.01, s = 100);
    ax.set_xlabel('x');
    ax.set_ylabel('y');
    ax.set_zlabel('z');

    plt.savefig(output_dir + event + '/' + event + '_conversion.png')
    plt.close()

############################ Conversion probabilities ###########################################
gag = 1. # In units of 10^{-14} GeV-1
add_to_readme_probs(event, gag)

if __name__ == '__main__':
    mp.freeze_support()
    ncores = mp.cpu_count() - local_run
    pool = mp.Pool(ncores)

    probabilities = conversion_probabilities_est(chosen_hits, gag, NS, pool)

data = []
for i, hit in enumerate(chosen_hits):
    data.append(np.append(hit, probabilities[i]))

np.save(output_dir + event + '/' + event + '_data.npy', data)

############################ Histogram of probabilities ###########################################

if draw_plots:
    plt.hist(np.log10(probabilities), bins=np.linspace(np.log10(np.min(probabilities)), np.log10(np.max(probabilities))));
    plt.xlabel('Log_10(Probability)')

    plt.savefig(output_dir + event + '/' + event + '_probs.png')

readme = open(output_dir + event + '/README.txt', 'a')
readme.write("Run Time: " + str(round((time.time() - start_time)/60, 2)) + " minutes.\n")

print(event)
print("Run Time: " + str(round((time.time() - start_time)/60, 2)) + " minutes.")
