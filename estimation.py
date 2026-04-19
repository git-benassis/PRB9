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
        # print(f"x={S},f(x)={val}")
        delta = P1_delta(S, r, sigma, T2_T1)
        deriv = - 1 + delta     

        if abs(val) < tol:
            return S
        
        S = S - val / deriv
    
    return S

def estimate_P2(S_values, K, Kbar, r, T1_idx, T2_idx, T1_val, T2_val):
    payoffs = []
    for S in S_values:
        if S[T1_idx] >= Kbar:
            payoff = np.exp(-r * T2_val) * max(K - S[T2_idx], 0)
        else:
            payoff = np.exp(-r * T1_val) * max(K - S[T1_idx], 0)
        payoffs.append(payoff)
    return np.array(payoffs) 

def IC2(S_values, K, Kbar, r, T1_idx, T2_idx, T1_val, T2_val, N):

    payoffs = estimate_P2(S_values, K, Kbar, r, T1_idx, T2_idx, T1_val, T2_val)
    
    mean_price = np.mean(payoffs)
    std = np.std(payoffs)
    
    error = 1.645 * std / np.sqrt(N)
    return [mean_price + error, mean_price - error, mean_price]

def IC2_antithetic(S_values, K, Kbar, r, T1_idx, T2_idx, T1_val, T2_val, N):

    payoffs = estimate_P2(S_values, K, Kbar, r, T1_idx, T2_idx, T1_val, T2_val)
    
    paired_payoffs = (payoffs[0::2] + payoffs[1::2]) / 2
    
    mean_price = np.mean(paired_payoffs)

    std = np.std(paired_payoffs)
    
    error = 1.645 * std / np.sqrt(len(paired_payoffs))
    return [mean_price + error, mean_price - error, mean_price]


from scipy.stats import norm
def Longstaff_Schwartz(S0,r,sigma,K,T1,T2,T3,N,n_steps):
    # à simuler N fois
    S=gen.multi_S_antithetic(T3, S0, r, sigma, N, n_steps)
    idx1 = int(round(T1 * n_steps / T3))  # 1/5 * 252 = 50
    idx2 = int(round(T2 * n_steps / T3))  # 3/5 * 252 = 151  
    
    S1 = S[:, idx1]  # Toutes trajectoires à T1
    S2 = S[:, idx2]  # Toutes à T2

    dt=T3-T2
    P1_T2_T3 = np.array([calculate_P1(r,sigma,K,s2,dt) for s2 in S2])
    # print(f"P1_T2_T3 min/max: {P1_T2_T3.min():.2f} / {P1_T2_T3.max():.2f}")
    # print(f"P1_T2_T3 mean: {P1_T2_T3.mean():.2f}")

    P2_at_T2= np.maximum(np.maximum(K-S2,0),P1_T2_T3)
    B=np.column_stack([np.ones(N), S1, S1**2, S1**3])
    omega, residuals, rank, sv = np.linalg.lstsq(B, P2_at_T2, rcond=None)

    # On resimule S[T1] en gardant les mêmes omega
    S_new=gen.multi_S_antithetic(T3,S0,r,sigma,N,n_steps)
    S1_new=S_new[:, idx1]
    B_new=np.column_stack([np.ones(N), S1_new, S1_new**2, S1_new**3])
    cont_T1 = np.exp(-r * (T2 - T1)) * (B_new @ omega)
    exer_T1 = np.maximum(K - S1_new, 0)

    P3_paths = np.exp(-r * T1) * np.maximum(exer_T1, cont_T1)
    return np.mean(P3_paths)

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

S0, r, sigma, K = 1.0, 0.02, 0.2, 1.0
T1, T2, T3 = 1.0, 3.0, 5.0  # Années
n_steps= 252 # jours/an
N_test = 100000

P3 = Longstaff_Schwartz(S0, r, sigma, K, T1, T2, T3, N_test,n_steps)
print(f"P3 ≈ {P3:.4f}")
