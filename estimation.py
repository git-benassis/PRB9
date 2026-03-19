import generation as gen
import numpy as np

def estimate_P1(num_trials, t,so,r,sigma):
    esp = 0
    for i in range(num_trials):
        esp+=max(gen.S(t,so,r,sigma),0)
    return np.exp(-r*t)*esp/num_trials
