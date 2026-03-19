import plot, estimation as est, generation as gen

n=1000
T=252*5
s0=1
r=0.02
sigma=0.2
K=1 

# S=gen.S(T,s0,r,sigma)
# plot.plot_brownian(S)

S_values=gen.multi_S(T,s0,r,sigma,n)
plot.plot_multi_S(S_values)
# print(est.estimate_P1(n,T,1,0.02,0.2))

# print(gen.multi_S(T,1,0.02,0.2,n))