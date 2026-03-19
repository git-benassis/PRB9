import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def generate_uniform(n, dim = 1):
    return npr.uniform(size=(n, dim))

def generate_gaussian(n, dim = 1):
    U1 = generate_uniform(n, dim)
    U2 = generate_uniform(n, dim)
    return np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)

def generate_brownian_motion(n, dim = 1):
    Increments=generate_gaussian(n)
    return np.concatenate((np.zeros(1),np.cumsum(Increments)))

T=1
n=100
W=generate_brownian_motion(n)
temps=np.linspace(0,T,n+1)
plt.plot(temps,W)
plt.show()