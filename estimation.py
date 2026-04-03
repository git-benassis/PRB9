import generation as gen
import numpy as np
from scipy.stats import norm

# estimation of an European call using Monte Carlo
def estimate_P1(S_values, K, r, T):
    esp = 0
    for S in S_values:
        esp += max(K - S[-1], 0)
    return np.exp(-r * T) * esp / len(S_values)

# calculate the analytical value of an European call
def calculate_P1(r,sigma,K,s0,T):
    d1 = (np.log(s0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*gen.repartition_gaussienne(-d2)-s0*gen.repartition_gaussienne(-d1)

# 5% Confidence Interval of P1 using the Monte Carlo estimator   
def IC(S,K, r, T, s0, sigma, N, affiche = False):
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

# value of Kbar for a 2 exercices Bermuda option
def estimate_K(S0, r, K, sigma, T2_T1, tol, max_iter):
    
    def P1_delta(S, r, sigma, T):
        d1 = (np.log(S0/S) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return (-gen.repartition_gaussienne(-d1))
    
    def f(S):
        return (K - S) - calculate_P1(r,sigma,S,S0,T2_T1)

    S = 0.9*K

    for _ in range(max_iter):
        val = f(S)
        print(f"x={S},f(x)={val}")
        delta = P1_delta(S, r, sigma, T2_T1)
        deriv = - 1 + delta     

        if abs(val) < tol:
            return S
        
        S = S - val / deriv
    
    return S

# price of a 2 exercices Bermuda option
def estimate_P2(S_values, K, Kbar, r, T1, T2):
    esp = 0
    for S in S_values:
        if S[T1] >= Kbar:
            esp += np.exp(-r*T2)*max(K - S(T2),0)
        else:
            esp += np.exp(-r*T1)*max(K - S(T1),0)
    return esp / len(S_values)

def IC2(S,K, Kbar, r, T1, T2, N, affiche = False):
    if(S[T1] >= Kbar):
        P = np.exp(-r*T2)*max(K - S(T2),0)
    else:
        P = np.exp(-r*T1)*max(K - S(T1),0)
    P2 = estimate_P2(S,K, Kbar,r,T1,T2)
    # 90% Confidence interval bounds
    std = np.std(P)
    error = 1.645*std/np.sqrt(N)
    CI_up = P2 + error
    CI_down = P2 - error
    if(affiche):
        print("Prix estimé :", P2)
        print("Confidence Interval up :", CI_up)
        print("Confidence Interval down :", CI_down)
        print("error :", error)
    return [CI_up, CI_down, error]

def Longstaff_Schwartz(S0,r,sigma,K,T1,T2,T3,N):
    # à simuler N fois
    S=gen.multi_S_antithetic(T3, S0, r, sigma, N)

    S1, S2, S3=S[T1], S[T2], S[T3]
    B=np.array([np.ones(N), S1, S1**2, S1**3])

    # Changer parce que la on calcule le payoff et pas vraiment le prix en T2 
    dt=T3-T2
    P2_at_T2= max(max(K-S2,0),calculate_P1(r,sigma,K,S2,dt))
    
    omega, residuals, rank, sv = np.linalg.lstsq(B, P2_at_T2, rcond=None)

    # On resimule S[T1] en gardant les mêmes omega
    S_new=gen.multi_S_antithetic(T3,S0,r,sigma,N)
    S1_new=S_new[T1]
    B_new=np.array([np.ones(N), S1_new, S1_new**2, S1_new**3])
    return(np.exp(-r*(T2-T1))* (B_new@omega))


# Paramètres

# N = 100000
# K = 1
# r = 0.02
# T = 5
# sigma = 0.2
# s0 = 1

# S = gen.multi_S(T,s0,r,sigma,N)
# S = np.array(S)
# IC(S,K,r,T,s0,N,True)