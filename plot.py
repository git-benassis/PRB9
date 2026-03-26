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