import plot, estimation as est, generation as gen

n=1000
T=252*5
s0=1
r=0.02
sigma=0.2
K=1 
# plot.plot_brownian(W)

S=gen.S(T,s0,r,sigma)
plot.plot_brownian(S)

# print(est.estimate_P1(n,T,1,0.02,0.2))

# print(gen.multi_S(T,1,0.02,0.2,n))