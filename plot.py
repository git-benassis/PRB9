import matplotlib.pyplot as plt
import numpy as np
import generation as gen
import estimation as est

def plot_brownian(n,T):
    W=gen.generate_brownian_motion(n)
    temps=np.linspace(0,T,n+1)
    plt.plot(temps,W)
    plt.show()

