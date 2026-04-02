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
    
    indices_n = list(range(2, nb_traj + 1, 2))
    
    for n in indices_n:
        S_sub = S_global[:n, :]
        S_anti_sub = S_global_anti[:n, :]
        
        # Calcul Classique
        res = est.IC2(S_sub, K, Kbar, r, idx1, idx2, T1_val, T2_val, n)
        IC_up.append(res[0])
        IC_down.append(res[1])
        P2_est.append(res[2])
        
        # Calcul Antithétique
        res_a = est.IC2_antithetic(S_anti_sub, K, Kbar, r, idx1, idx2, T1_val, T2_val, n)
        IC_anti_up.append(res_a[0])
        IC_anti_down.append(res_a[1])
        P2_est_anti.append(res_a[2])
        
    plt.figure(figsize=(10,6))
    
    plt.plot(indices_n, P2_est, 'r-', label='P2 Classique')
    plt.plot(indices_n, P2_est_anti, 'g-', label='P2 Antithétique')
    
    plt.plot(indices_n, IC_up, 'orange', linestyle='--', alpha=0.5, label='IC 90% Classique')
    plt.plot(indices_n, IC_down, 'orange', linestyle='--', alpha=0.5)
    
    plt.plot(indices_n, IC_anti_up, 'b', linestyle='--', alpha=0.5, label='IC 90% Antithétique')
    plt.plot(indices_n, IC_anti_down, 'b', linestyle='--', alpha=0.5)
    
    plt.xlabel("Nombre de trajectoires (N)") # Ce n'est plus log(N) ici, mais N réel
    plt.ylabel("Prix de l'option P2")
    plt.legend()
    plt.title("Convergence et réduction de variance (Antithétique vs Classique)")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_P2_fonction_S0(S_global,K,Kbar,r,idx_T1,idx_T2,T1,T2,Val_S0,sigma, m, N):
    T_max = np.maximum(T1,T2)
    Val_P2 = []
    Val_P1_T1 = []
    Val_P1_T2 = []

    for S0_alt in Val_S0:
        S_global = gen.multi_S_antithetic(T_max, S0_alt, r, sigma, m, N)
        P2 = np.mean(est.estimate_P2(S_global, K, Kbar, r, idx_T1, idx_T2, T1, T2))
        Val_P2.append(P2)
        P1_T1 = est.estimate_P1(S_global,K,r,T1)
        P1_T2 = est.estimate_P1(S_global,K,r,T2)
        Val_P1_T1.append(P1_T1)
        Val_P1_T2.append(P1_T2)

    plt.figure(figsize=(12, 7))
    
    plt.plot(Val_S0, Val_P2, color='red', linewidth=2, label=f'P2 : Option Composée (K={K}, Kbar={Kbar:.2f})')
    plt.plot(Val_S0, Val_P1_T1, color='blue', linestyle='--', alpha=0.8, label=f'P1 : Put Européen à T={T1}')
    plt.plot(Val_S0, Val_P1_T2, color='green', linestyle=':', alpha=0.8, label=f'P1 : Put Européen à T={T2}')
    
    plt.title(f"Sensibilité du prix des options à S0\n(Simulation Monte Carlo Antithétique, m={m} trajectoires)", fontsize=14)
    plt.xlabel("Prix de l'actif sous-jacent à t=0 (S0)", fontsize=12)
    plt.ylabel("Prix de l'option (Valeur actualisée)", fontsize=12)
    
    plt.axvline(x=K, color='black', linestyle='-', alpha=0.2, label='Strike K')
    
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.show()