import numpy as np
import string
import random
import os

G_N = 1.325e11  # Newton's constant in km^3/M_Sun/s^2
c = 2.99792458*1e5  # Speed of light in km/s
rho_eq = 5.78*1e-28 # Energy density at matter radiation equality in units of 10^{-10}*M_Sun/km^3
output_dir = 'C:/Users/gsali/Dropbox/output/axionns/'
conv_factor_eV_GHz = 1.5192669e6
conv_factor_km_eVinv = 1.e10/1.9732705
conv_factor_G_eV2 = 1/14.440271

# Magnitude of vectors
def mag_vector(vs):
    return np.sqrt(sum(np.power(np.array(vs).T, 2)))

# Linear combination of vectors with coefficients given by numbers
def numbers_times_vectors(numbers, vectors):
    return np.multiply(vectors.T, numbers).T

# Method to draw uniformily distributed points along the unit sphere (Marsaglia 1972)
def sphere_point_picking():
    x1, x2 = 1, 1
    xnorm = np.power(x1,2) + np.power(x2,2)
    while xnorm >= 1:
        x1, x2 = np.random.uniform(-1,1,2)
        xnorm = np.power(x1,2) + np.power(x2,2)

    nx = 2*x1*np.sqrt(1 - xnorm)
    ny = 2*x2*np.sqrt(1 - xnorm)
    nz = 1 - 2*xnorm

    return np.array([nx, ny, nz])

def angle_between_vecs(v1, v2):  # Find the angle between two 1d numpy arrays
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def mkdir_event(NS, clump, r_in, v_in, n_in, size=6, chars=string.ascii_uppercase):     # Variable 'clump' should be an instance of AS or MC
    id_str = ''.join(random.choice(chars) for _ in range(size))
    event_name = clump.clump_type_short + id_str
    os.makedirs(output_dir + event_name, exist_ok=True)

    f = open(output_dir + event_name + '/' + 'README.txt', 'w')

    f.write('Event name: ' + event_name + '\n')
    f.write('Axion mass: ' + str(clump.axionmass) + ' x 10^{-5} eV' + '\n\n')

    f.write('-'*25 + ' Neutron star properties ' + '-'*50 + '\n')
    f.write('Mass: ' + str(NS.mass) + ' M_Sun\n')
    f.write('Radius: ' + str(NS.radius) + ' km\n')
    f.write('Period: ' + str(NS.period) + ' s\n')
    f.write('Axis of rotation: ' + str(NS.axis) + '\n')
    f.write('Magnetic dipole moment: ' + str(NS.dipole_moment) + ' x 10^30 Am^2\n\n')

    f.write('-'*25 + ' Axion clump properties ' + '-'*50 + '\n')
    f.write('Clump type: ' + clump.clump_type + '\n')
    f.write('Velocity dispersion curve: ' + str(clump.vdisp_type) + '\n')
    if clump.clump_type == 'Dilute axion star':
        f.write('Mass: ' + str(clump.mass) + ' x 10^{-12} M_Sun\n')
        f.write('Radius: ' + '{:.2e}'.format(clump.radius99()) + ' km\n')
    elif clump.clump_type == 'Axion Minicluster (NFW profile)':
        f.write('Mass: ' + str(clump.mass) + ' x 10^{-10} M_Sun\n')
        f.write('Truncation radius: ' + '{:.2e}'.format(clump.radius_trunc()) + ' km\n')
        f.write('Delta: ' + '{:.2e}'.format(clump.delta) + '\n')
        f.write('Concentration: ' + '{:.2e}'.format(clump.concentration) + '\n')
    f.write('Roche radius: ' + '{:.2e}'.format(clump.roche(NS)) + ' km\n')
    f.write('Initial position: ' + '[ ' + '{:.2e}'.format(np.array(r_in[0])) + ' ' + '{:.2e}'.format(np.array(r_in[1])) + ' ' + '{:.2e}'.format(np.array(r_in[2])) + ' ] km\n')
    f.write('Initial velocity: ' + '[ ' + '{:.2e}'.format(np.array(v_in[0])) + ' ' + '{:.2e}'.format(np.array(v_in[1])) + ' ' + '{:.2e}'.format(np.array(v_in[2])) + ' ] km/s\n\n' )

    f.write('-'*25 + ' Trajectories ' + '-'*50 + '\n')
    
    f.close()

    return event_name

def join_npys(directory_str):
    directory = os.fsencode(output_dir + directory_str)
    data_all, last_tag = [], 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npy"):
            data_batch = np.load(output_dir + directory_str + '/' + filename)
            data_batch = np.array([data_batch_elem + np.array([last_tag, 0, 0, 0, 0, 0, 0, 0]) for data_batch_elem in data_batch])
            last_tag = data_batch[-1][0] + 1
            data_all.append(data_batch)
            os.remove(output_dir  + directory_str + '/' + filename)
    
    data_array = np.concatenate(data_all)
    np.save(output_dir + directory_str + '/' + directory_str, data_array)