import plot, estimation as est, generation as gen
import matplotlib.pyplot as plt
import numpy as np
import test

# Paramètres

N = 10000
K = 1
r = 0.02
T = 5
T1 = 3
T2 = 5
sigma = 0.2
s0 = 1


# S=gen.S(T,s0,r,sigma)
# plot.plot_brownian(S)

#S_values=gen.multi_S(T,s0,r,sigma,N)
#plot.plot_multi_S(S_values)
# print(est.estimate_P1(n,T,1,0.02,0.2))

# print(gen.multi_S(T,1,0.02,0.2,N))


# S = gen.multi_S(T,s0,r,sigma,N)
# S = np.array(S)
# nb_traj = np.logspace(1,5, num=20, dtype=int)
# plot.plot_P1(S,K,r,T,s0,sigma,N,nb_traj)

Kbar=est.estimate_K(1,0.02,1.0,0.2,3.0,1e-06,100)
print(Kbar)

S_T1 = 1.0  # Valeur typique à T1
K, r, sigma, dt = 1.0, 0.02, 0.2, 3.0  # T2-T1 = 2 ans
Kstar = test.estimate_Kstar(S_T1, K, r, sigma, dt)
print(f"K* ≈ {Kstar:.4f}")  # ~0.9342

# Plot Q10

n_steps = 100
idx_T1 = 60
idx_T2 = n_steps
Val_S0 = np.linspace(0.5, 2.0, 50) 
Kbar_q10 = est.estimate_K(s0, r, K, sigma, T2-T1, 1e-6, 100)

plot.plot_P2_fonction_S0(None,K,Kbar_q10,r,idx_T1,idx_T2,T1,T2,Val_S0,sigma,50000,n_steps)

# switch option

S0_vals = np.linspace(0.5, 2.0, 200)
sigmas = [0.1, 0.2, 0.4]
sw_results = est.calculate_switch_option(S0_vals, r, sigmas, K, T1, T2, N)

# Plot Q11
plot.plot_SW(S0_vals,sigmas,sw_results)