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

def S(t,so,r,sigma):
    W = generate_brownian_motion(t)
    return so*np.exp((r-sigma**2/2)*t+sigma*W[-1])

def repartition_gaussienne(x): # Fonction de répartition d'une gaussienne centrée réduite
    t = 1/(1+b[0]*x)
    b = [0.2316419,0.319381530,-0.356563782,1.781477937,-1.821255978,1.330274429]
    return 1/(np.sqrt(2*np.pi))*np.exp(-0.5*x**2)*(b[1]*t+b[2]*t**2+b[3]*t**3+b[4]*t**4+b[5]*t**5)