import generation as gen
import numpy as np

def estimate_P1(num_trials, t,so,r,sigma):
    esp = 0
    for i in range(num_trials):
        esp+=max(gen.S(t,so,r,sigma),0)
    return np.exp(-r*t)*esp/num_trials

def calculate_P1(r,sigma,K,s0,T):
    d1 = (np.log(s0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*gen.repartition_gaussienne(-d2)-s0*gen.repartition_gaussienne(-d1)