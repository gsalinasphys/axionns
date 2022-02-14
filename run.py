import numpy as np
import matplotlib.pyplot as plt
from classes.ns import NeutronStar
import multiprocess as mp
from classes.axionstar import AxionStar
from classes.minicluster import AxionMiniclusterNFW
from classes.particles import Particles
from scripts.orbits import evolve
from scripts.basic_functions import output_dir, conv_factor_Msolar_eV
from scripts.conversion import find_all_hits, divide_into_singles

plt.rcParams['figure.figsize'] = [24, 16]

def choose_clump(Mass, vy_in, b, MC_or_AS = 'MC', vdisp_type='Maxwell-Boltzmann', delta = 1.55, concentration = 100):    # Mass in 10^{-10} M_solar for MC, 10^{-12} M_solar for AS 
    if MC_or_AS == 'MC':
        AC = AxionMiniclusterNFW(Mass, Particles.axionmass, vdisp_type=vdisp_type, delta = delta, concentration=concentration)
        AC.center, AC.vcenter = [b*AC.radius_trunc(), 1.e16, 0], [0, vy_in, 0]
    elif MC_or_AS == 'AS':
        if vdisp_type=='Maxwell-Boltzmann':
            vdisp_type = None
        AC = AxionStar(Mass, Particles.axionmass, vdisp_type=vdisp_type)
        AC.center, AC.vcenter = [b*AC.radius99(), 1.e16, 0], [0, vy_in, 0]

############################## Define parameters here ###################################

M_NS, R_NS = 1, 10

MC_or_AS = 'MC'
Mass = 1.   # In units of 10^{-10} M_solar or 10^{-12} M_solar
vy_in, b = -200., 0.2   # b in units of MC.radius_trunc() or AS.radius99()
delta, concentration = 1.55, 100

length, nparticles, batch_size, conservation_check = np.array([-1,1])/300, int(1.e3), int(1.e2), False

############################ Compute trajectories ###########################################

chosen_clump = choose_clump(Mass, vy_in, b, MC_or_AS = MC_or_AS, delta = delta, concentration = concentration)
NS = NeutronStar(M_NS, R_NS)

if __name__ == '__main__':
    mp.freeze_support()
    ncores = mp.cpu_count()
    pool = mp.Pool(ncores)

    event, conservation_checks, mass_in, total_drawn = evolve(NS, chosen_clump, pool, nparticles, length = length, batch_size=batch_size, conservation_check = conservation_check)

###################### Number of axions per trajectory ######################################

Ntotal_drawn = total_drawn/mass_in*chosen_clump.mass
if MC_or_AS == 'MC':
    Ntotal = 1e-5*conv_factor_Msolar_eV*chosen_clump.mass/Particles.axionmass
elif MC_or_AS == 'AS':
    Ntotal = 1e-7*conv_factor_Msolar_eV*chosen_clump.mass/Particles.axionmass

axions_per_traj = Ntotal/Ntotal_drawn

part_trajs = np.load(output_dir + event + '/' + event + '.npy')
nparticles = int(part_trajs[-1][0]) + 1

############################ Write to README ###########################################

readme = open(output_dir + event + '/README.txt', 'a')
readme.write('Number of saved trajectories: ' + str(nparticles) + '\n')
if conservation_check:
    readme.write('Energy and angular momentum conservation was checked. It is valid up to ' + str(np.round(np.max(conservation_checks), 2)) + ' percent.\n')
else:
    readme.write('Energy and angular momentum conservation was not checked.\n')
readme.write('Each trajectory correponds to ' + '{:.2e}'.format(axions_per_traj) + ' axions.\n')
readme.write('Axions were drawn from a cylinder of length ' + '{:.2e}'.format((length[1] - length[0])*chosen_clump.radius_trunc()) + ' km centered at ' + '{:.2e}'.format((length[1] + length[0])*chosen_clump.radius_trunc()) + ' km from the projection of the clump\'s center on the y-axis.')
readme.close()

############################ Plot some trajectories ###########################################

ax = plt.gca()
ax.set_aspect('equal')
nsamples = np.min([int(1e3), nparticles])
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

############################ Conversion in the NS magnetosphere ###########################################

single_particles = divide_into_singles(part_trajs)

if __name__ == '__main__':
    all_hits = find_all_hits(single_particles, NS, pool)

np.save(output_dir + event + '/' + event + '_conversion.npy', all_hits)

all_ts = np.array(all_hits).T[1]
all_ts -= np.min(all_ts)
all_ts /= 3600.

chosen_hour = int(all_ts[-1])
indices = np.where(np.logical_and(np.sort(all_ts) > chosen_hour, np.sort(all_ts) < chosen_hour + 1))[0]
chosen_hits = all_hits[indices[0]: indices[-1]]

############################ Plot conversion surface ###########################################

X, Y, Z = np.array(chosen_hits).T[2:5]
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, s = 1);

Xsurf, Ysurf, Zsurf = NS.conversion_surface_est(0., Particles.axionmass).T
ax.scatter3D(Xsurf, Ysurf, Zsurf, c = 'purple', alpha = 0.01);
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('z');

plt.savefig(output_dir + event + '/' + event + '_conversion.png')