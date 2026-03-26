import plot, estimation as est, generation as gen
import numpy as np

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


S = gen.multi_S(T,s0,r,sigma,N)
S = np.array(S)
nb_traj = np.logspace(1,5, num=20, dtype=int)
plot.plot_P1(S,K,r,T,s0,sigma,N,nb_traj)