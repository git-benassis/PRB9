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

## Methodology

### European Put $P_1$
Closed-form Black-Scholes formula:

$$P_1 = e^{-rT}\mathbb{E}\left[(K - S(T))_+\right]$$

### Bermudan Put $P_2$ (2 exercise dates)
Optimal stopping at $T_1$: exercise if $S(T_1) < \bar{K}$, where $\bar{K}$ satisfies:

$$P_2 = e^{-rT_2}\mathbb{E}\left[(K - S(T_2))_+ \mathbf{1}_{S(T_1) \geq \bar{K}}\right] + e^{-rT_1}\mathbb{E}\left[(K - S(T_1))\mathbf{1}_{S(T_1) < \bar{K}}\right]$$

### Bermudan Put $P_3$ (3 exercise dates) — Longstaff-Schwartz
Since $P_2$ at $T_1$ has no closed form, the Longstaff-Schwartz algorithm approximates it by regressing $P_2|_{T_2}$ onto the polynomial basis $\mathcal{B} = \{1, S_{T_1}, S_{T_1}^2, S_{T_1}^3\}$.

The regression coefficients $\{\omega_k\}$ solve:

$$A \cdot \Omega = B$$

where $A_{i,j} = \widehat{\text{Cov}}(S_{T_1}^i, S_{T_1}^j)$ and $B_i = \widehat{\text{Cov}}(P_2|_{T_2}, S_{T_1}^i)$.

Fresh trajectories are then re-simulated to estimate $P_3$ without measurability bias.

***

## Implementation Notes

### Random Number Generation
All simulations use a **uniform generator** as the sole source of randomness, transformed via the **Abramowitz-Stegun approximation** of $\Phi$:

$$\Phi(x) \approx 1 - \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} \left( b_1 t + b_2 t^2 + b_3 t^3 + b_4 t^4 + b_5 t^5 \right), \quad t = \frac{1}{1 + b_0 x}$$

### Variance Reduction
Antithetic variates are used throughout: for each path driven by $Z \sim \mathcal{N}(0,1)$, its counterpart $-Z$ is also simulated.

### Bias-Free Longstaff-Schwartz
The regression step and the valuation step use **two independent sets of trajectories** to avoid the non-measurability bias that arises from reusing the same paths.

***

## Dependencies

```
numpy
scipy
matplotlib
```

***

## Usage

```bash
python main.py
```

Or use individual modules:

```python
from estimation import calculate_P1, Longstaff_Schwartz

# European put (closed-form)
P1 = calculate_P1(r=0.02, sigma=0.2, K=1, S0=1, T=5)

# Bermudan put P3 via Longstaff-Schwartz
P3 = Longstaff_Schwartz(S0=1, r=0.02, sigma=0.2, K=1,
                         T1=1, T2=3, T3=5, N=50000, n_steps=252)
```
