import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# create an array of n float in [0,1) 
def generate_uniform(n, dim = 1):
    return npr.uniform(size=(n, dim))

# create an array of n random number following N(0,1)
def generate_gaussian(n, dim = 1):
    U1 = generate_uniform(n, dim)
    U2 = generate_uniform(n, dim)
    return np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)

# simulate a discrete Brownian motion path of size n+1
def generate_brownian_motion(n_steps, T_max):
    dt = T_max / n_steps
    increments = npr.normal(0, np.sqrt(dt), n_steps) 
    return np.concatenate((np.zeros(1), np.cumsum(increments)))

# simulate the array of S(t) for each t in [1,T]
def S(T,so,r,sigma):
    W = generate_brownian_motion(T)
    S = []
    for t in range(T):
        S.append(so*np.exp((r-sigma**2/2)*t+sigma*W[t+1]))
    return S

# generate m independant vector S
def multi_S(T_max, s0, r, sigma, m, n_steps):
    paths = []
    t_axis = np.linspace(0, T_max, n_steps + 1)
    drift_coeff = (r - 0.5 * sigma**2)
    
    for _ in range(m):
        W = generate_brownian_motion(n_steps, T_max)
        S_path = s0 * np.exp(drift_coeff * t_axis + sigma * W)
        paths.append(S_path)
    return np.array(paths)

# generate m independant vector S using antithetic variables
def multi_S_antithetic(T_max, s0, r, sigma, m, n_steps):
    paths = []
    t_axis = np.linspace(0, T_max, n_steps + 1)
    drift = (r - 0.5 * sigma**2)
    
    for _ in range(int(m / 2)):
        W_plus = generate_brownian_motion(n_steps, T_max) 
        W_minus = -W_plus

        S_plus = s0 * np.exp(drift * t_axis + sigma * W_plus)
        S_minus = s0 * np.exp(drift * t_axis + sigma * W_minus)
        
        paths.append(S_plus)
        paths.append(S_minus)
    return np.array(paths)

# Repartition function of N(0,1)
def repartition_gaussienne(x): 
    b = [0.2316419,0.319381530,-0.356563782,1.781477937,-1.821255978,1.330274429]
    t = 1/(1+b[0]*x)
    return 1 - 1/(np.sqrt(2*np.pi))*np.exp(-0.5*x**2)*(b[1]*t+b[2]*t**2+b[3]*t**3+b[4]*t**4+b[5]*t**5)

