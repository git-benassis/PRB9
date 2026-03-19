import generation as gen
import numpy as np

def estimate_P1(num_trials, r,T):
    esp = 0
    for i in range(num_trials):
        esp+=max(gen.S(),0)
    return np.exp(-r*T)*esp/num_trials

