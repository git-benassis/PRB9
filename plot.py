import matplotlib.pyplot as plt
import numpy as np
import generation as gen
import estimation as est

def plot_brownian(W):
    plt.plot(W)
    plt.show()

def plot_multi_S(S_values):
    for S in S_values:
        plt.plot(S)
    plt.show()