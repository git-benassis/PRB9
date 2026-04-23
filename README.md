# Bermudan Options Pricing — Black-Scholes & Longstaff-Schwartz

> Cours PRB222 — ENSTA Paris / Institut Polytechnique de Paris  
> Monte Carlo pricing of Bermudan put options under Black-Scholes dynamics

***

## Overview

This project implements the pricing of **Bermudan put options** in a Black-Scholes framework, progressively from a simple European put to a 3-exercise Bermudan option using the **Longstaff-Schwartz regression algorithm**.

A Bermudan option gives its holder the right to sell an asset at a fixed strike price $K$, but only at a finite set of predefined dates $\{T_1, T_2, T_3\}$. The key challenge is determining the **optimal exercise policy** at each date.

***

## Mathematical Setting

The underlying asset follows Black-Scholes dynamics under the risk-neutral measure:

$$dS(t) = S(t)(r \, dt + \sigma \, dW(t)), \quad S(0) = S_0 > 0$$

**Default parameters:**

| Parameter | Value |
|-----------|-------|
| $S_0$ | 1.0 |
| $r$ (risk-free rate) | 0.02 |
| $\sigma$ (volatility) | 0.2 |
| $K$ (strike) | 1.0 |
| $T_1$ | 1 year |
| $T_2$ | 3 years |
| $T_3$ | 5 years |

***

## Project Structure

```
bermudan-options/
├── generation.py      # Random number generation (Uniform → Gaussian → GBM paths)
├── estimation.py      # Pricing functions: P1, P2, P3, Longstaff-Schwartz
├── plot.py            # Plotting utilities
├── main.py            # Main script: runs all pricing and generates plots
└── README.md
```

### `generation.py`
Implements all random number generation from scratch using only a **uniform random generator**, as required:
- Gaussian generation via the Box-Muller or inverse CDF method
- Gaussian CDF approximation via Abramowitz-Stegun formula (error $< 7.5 \times 10^{-8}$)
- Simulation of GBM paths $\{S_{T_1}, S_{T_2}, S_{T_3}\}$
- Antithetic variates for variance reduction

### `estimation.py`
Contains all pricing functions:
- `calculate_P1` — closed-form Black-Scholes European put price
- `estimate_K` — numerical computation of the optimal exercise threshold $\bar{K}$ for $P_2$
- `estimate_P2` — Monte Carlo estimator of the Bermudan 2-exercise put price
- `Longstaff_Schwartz` — regression-based Monte Carlo estimator of the Bermudan 3-exercise put price $P_3$

### `plot.py`
Plotting utilities for all figures:
- Convergence plots with 90% confidence intervals
- $P_1$, $P_2$, $P_3$ as a function of $S_0$
- Regression quality diagnostics for Q16

### `main.py`
Entry point that runs all questions sequentially and produces the figures.

***
## Usage

```bash
python main.py
```
