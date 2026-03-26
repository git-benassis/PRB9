import plot, estimation as est, generation as gen
import numpy as np
import test

# Paramètres

N = 100000
K = 1
r = 0.02
T = 5
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