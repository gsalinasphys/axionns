import numpy as np

G_N = 1.325e11  # Newton's constant in km^3/Msol/s^2

# Magnitude of vectors
def mag_vector(vs):
    return np.sqrt(sum(np.power(np.array(vs).T, 2)))

def numbers_times_vectors(numbers, vectors):
    return np.multiply(vectors.T, numbers).T