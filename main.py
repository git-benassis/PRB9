import plot, estimation as est, generation as gen
import numpy as np
import test

# Paramètres

T1 = 3
T2 = 5
T_max = 5
S0 = 1
r = 0.02
sigma = 0.2
m = 500
N = 1000 
K = 1


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

#Kbar=est.estimate_K(1,0.02,1.0,0.2,3.0,1e-06,100)
#print(Kbar)

#S_T1 = 1.0  # Valeur typique à T1
#K, r, sigma, dt = 1.0, 0.02, 0.2, 3.0  # T2-T1 = 2 ans
#Kstar = test.estimate_Kstar(S_T1, K, r, sigma, dt)
#print(f"K* ≈ {Kstar:.4f}")  # ~0.9342

Kbar = est.estimate_K(S0, r, K, sigma, T1, 1e-06, 100)
print(Kbar)

S_global = gen.multi_S(T_max, S0, r, sigma, m, N)
S_global_antithetic = gen.multi_S_antithetic(T_max, S0, r, sigma, m, N)

idx_T1 = int(T1 * (N / T_max))
idx_T2 = int(T2 * (N / T_max))

# plot.plot_P2(S_global,S_global_antithetic,K,Kbar,r,idx_T1,idx_T2,T1,T2,S0,sigma,m) # Q.9

# Q.10

Val_S0 = np.linspace(0.5,2,20)

plot.plot_P2_fonction_S0(S_global,K,Kbar,r,idx_T1,idx_T2,T1,T2,Val_S0,sigma, m, N)