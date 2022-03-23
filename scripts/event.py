import numpy as np

from scripts.basic_functions import output_dir, conv_factor_Msolar_eV

from classes.particles import Particles
from classes.axionstar import AxionStar
from classes.minicluster import AxionMiniclusterNFW

def choose_clump(Mass, vy_in, b, MC_or_AS = 'MC', vdisp_type='Maxwell-Boltzmann', delta = 1.55, concentration = 100):    # Mass in 10^{-10} M_solar for MC, 10^{-12} M_solar for AS, b in units of MC.radius_trunc() or AS.radius99()
    if MC_or_AS == 'MC':
        AC = AxionMiniclusterNFW(Mass, Particles.axionmass, vdisp_type=vdisp_type, delta = delta, concentration=concentration)
        AC.center, AC.vcenter = [b*AC.radius_trunc(), 1.e16, 0], [0, vy_in, 0]
    elif MC_or_AS == 'AS':
        if vdisp_type=='Maxwell-Boltzmann':
            vdisp_type = None
        AC = AxionStar(Mass, Particles.axionmass, vdisp_type=vdisp_type)
        AC.center, AC.vcenter = [b*AC.radius99(), 1.e16, 0], [0, vy_in, 0]
    return AC
        
def axions_per_traj(chosen_clump, total_drawn, mass_in):
    Ntotal_drawn = total_drawn/mass_in*chosen_clump.mass
    if chosen_clump.clump_type_short == 'MCNFW':
        Ntotal = 1e-5*conv_factor_Msolar_eV*chosen_clump.mass/Particles.axionmass
    elif chosen_clump.clump_type_short == 'dAS':
        Ntotal = 1e-7*conv_factor_Msolar_eV*chosen_clump.mass/Particles.axionmass

    return Ntotal/Ntotal_drawn

def add_to_readme(event, part_trajs, chosen_clump, axions_per_traj, conservation_check, conservation_check_result, length):
    nparticles = int(part_trajs[-1][0]) + 1

    readme = open(output_dir + event + '/README.txt', 'a')
    if chosen_clump.clump_type_short == 'MCNFW':
        readme.write('Sampling cylinder length: ' + '{:.2e}'.format((length[1] - length[0])*chosen_clump.radius_trunc()))
        readme.write('Sampling cylinder centered at ' + '{:.2e}'.format((length[1] + length[0])*chosen_clump.radius_trunc()) + ' km from the projection of the clump\'s center on the y-axis.\n\n')
    readme.write('Number of saved trajectories: ' + str(nparticles) + '\n')
    if conservation_check:
        readme.write('Energy and angular momentum conservation was checked. It is valid up to ' + str(np.round(np.max(conservation_check_result), 2)) + ' percent.\n')
    else:
        readme.write('Energy and angular momentum conservation was not checked.\n')
    readme.write('Each trajectory correponds to ' + '{:.2e}'.format(axions_per_traj) + ' axions.\n\n')

    readme.close()
        
def add_to_readme_probs(event, gag):
    readme = open(output_dir + event + '/README.txt', 'a')
    readme.write('-'*25 + ' Conversion Probabilities ' + '-'*50 + '\n')
    readme.write('Axion-photon coupling: ' +  '{:.2e}'.format(gag) + ' x 10^{-14} GeV^{-1}\n\n')
