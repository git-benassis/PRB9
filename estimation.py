import generation as gen
import numpy as np

def estimate_P1(S_values, K, r, T):
    esp = 0
    for S in S_values:
        esp += max(K - S[-1], 0)
    return np.exp(-r * T) * esp / len(S_values)

def calculate_P1(r,sigma,K,s0,T):
    d1 = (np.log(s0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*gen.repartition_gaussienne(-d2)-s0*gen.repartition_gaussienne(-d1)

def IC(S,K, r, T, s0, N, affiche = False):
    P = np.exp(-r * T) * np.maximum(K - S[:, -1], 0)
    P1 = estimate_P1(S,K,r,T)
    # 90% Confidence interval bounds
    std = np.std(P)
    error = 1.645*std/np.sqrt(N)
    CI_up = P1 + error
    CI_down = P1 - error
    if(affiche):
        print("Vrai prix :", calculate_P1(r,sigma,K,s0,T))
        print("Prix estimé :", P1)
        print("Confidence Interval up :", CI_up)
        print("Confidence Interval down :", CI_down)
        print("error :", error)
    return [CI_up, CI_down, error]

# Paramètres

N = 100000
K = 1
r = 0.02
T = 5
sigma = 0.2
s0 = 1

S = gen.multi_S(T,s0,r,sigma,N)
S = np.array(S)
IC(S,K,r,T,s0,N,True)