import numpy as np
import numpy.random as npr

def generate_uniform(n, dim = 1):
    return npr.uniform(size=(n, dim))

def generate_gaussian(n, dim = 1):
    U1 = generate_uniform(n, dim)
    U2 = generate_uniform(n, dim)
    return np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)

def generate_brownian_motion(n, dim = 1):
    return np.cumsum(generate_gaussian(n, dim), axis=0)