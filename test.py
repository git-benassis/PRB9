import numpy as np
from scipy.stats import norm

def black_scholes_put(S, K, r, sigma, T):
    """Prix put européen BS"""
    if T == 0 or S <= 0:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def put_delta(S, K, r, sigma, T):
    """Delta put = -N(-d1)"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return -norm.cdf(-d1)

def estimate_Kstar(S_T1, K, r, sigma, T2_T1, tol=1e-6, max_iter=50):
    """Newton pour f(x) = (K-x) - P_put(x,K,r,σ,T2-T1) = 0"""
    
    def f(x):
        return (K - x) - black_scholes_put(S_T1, x, r, sigma, T2_T1)
    
    def df(x):
        return -1 + put_delta(S_T1, x, r, sigma, T2_T1)
    
    # Initialisation intelligente
    x = 0.9 * K  # K* < K
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(fx) < tol:
            print(f"Convergence en {i} itérations: K* = {x:.6f}")
            return x
        
        if abs(dfx) < 1e-10:  # Dérivée trop petite
            print("Échec: dérivée trop faible")
            return None
            
        x_new = x - fx / dfx
        if x_new <= 0 or x_new >= K:
            print("Sortie du domaine [0,K]")
            return None
            
        print(f"Iter {i}: x={x:.6f}, f(x)={fx:.2e}")
        x = x_new
    
    print("Max iterations atteint")
    return x