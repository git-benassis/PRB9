import plot, estimation as est

n=10 
T=252*5 
# plot.plot_brownian(n,T)

print(est.estimate_P1(n,T,1,0.02,0.2,1))
