import matplotlib.pyplot as plt
import numpy as np
import generation as gen
import estimation as est

def plot_brownian(W):
    plt.plot(W)
    plt.show()

def plot_multi_S(S_values):
    for S in S_values:
        plt.loglog(S)
    plt.show()

def plot_P1(S_global, K, r, T, s0, sigma,N,nb_traj):
    P1_vrai = [est.calculate_P1(r, sigma, K, s0, T) for _ in range(len(nb_traj))]
    P1_est = []
    IC_up = []
    IC_down = []
    for N in nb_traj:
        S = S_global[:N,:]
        P1 = est.estimate_P1(S,K,r,T)
        P1_est.append(P1)
        CI = est.IC(S,K,r,T,s0,sigma,N)
        IC_up.append(CI[0])
        IC_down.append(CI[1])
    plt.plot(P1_vrai, color='red', label='Vrai Prix')
    plt.plot(P1_est, color = 'green', label='Prix estimé')
    plt.plot(IC_up, color = 'blue', label = 'Intervalle de confiance à 90%')
    plt.plot(IC_down, color = 'blue')
    plt.legend()
    plt.xlabel("Nombre de trajectoires (log(N))")
    plt.ylabel("Prix")
    plt.show()

def plot_P2(S_global, S_global_anti, K, Kbar, r, idx1, idx2, T1_val, T2_val, s0, sigma, nb_traj):
    P2_est, P2_est_anti = [], []
    IC_up, IC_down = [], []
    IC_anti_up, IC_anti_down = [], []
    
    # On commence à 2 pour pouvoir calculer un écart-type
    for n in range(2, nb_traj + 1):
        # Sélections des n premières trajectoires
        S_sub = S_global[:n, :]
        S_anti_sub = S_global_anti[:n, :]
        
        # Calcul pour Monte Carlo Classique
        res = est.IC2(S_sub, K, Kbar, r, idx1, idx2, T1_val, T2_val, n)
        IC_up.append(res[0])
        IC_down.append(res[1])
        P2_est.append(res[2])
        
        # Calcul pour Antithétique
        res_a = est.IC2(S_anti_sub, K, Kbar, r, idx1, idx2, T1_val, T2_val, n)
        IC_anti_up.append(res_a[0])
        IC_anti_down.append(res_a[1])
        P2_est_anti.append(res_a[2])
        
    plt.figure(figsize=(10,6))
    plt.plot(P2_est, 'r-', label='P2 Classique')
    plt.plot(P2_est_anti, 'g-', label='P2 Antithétique')
    plt.plot(IC_up, 'orange', linestyle='--', alpha=0.5, label='IC 90% Classique')
    plt.plot(IC_down, 'orange', linestyle='--', alpha=0.5)
    plt.plot(IC_anti_up, 'b', linestyle='--', alpha=0.5, label='IC 90% Antithétique')
    plt.plot(IC_anti_down, 'b', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title("Convergence de l'estimateur P2")
    plt.show()