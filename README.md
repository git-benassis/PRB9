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
├── generators.py          # Random number generation (Uniform → Gaussian → GBM paths)
├── estimators.py          # Pricing functions: P1, P2, P3, Longstaff-Schwartz
├── main.ipynb             # Full notebook with results and plots
└── README.md
```

***

## Key Results

### Q4 — European Put $P_1$ convergence
Monte Carlo estimation of $P_1 = e^{-rT}\mathbb{E}[(K - S(T))_+]$ with 90% confidence intervals, compared to the Black-Scholes closed-form price.

### Q9/Q10 — Bermudan Put $P_2$ (2 exercise dates)
The price $P_2$ is computed via:

$$P_2 = e^{-rT_2}\mathbb{E}\left[(K - S(T_2))_+ \mathbf{1}_{S(T_1) \geq \bar{K}}\right] + e^{-rT_1}\mathbb{E}\left[(K - S(T_1))\mathbf{1}_{S(T_1) < \bar{K}}\right]$$

where $\bar{K}$ is the optimal exercise threshold, computed numerically.  
Variance reduction via **antithetic variates** is implemented and compared to the classical estimator.

### Q14/Q15 — Bermudan Put $P_3$ (3 exercise dates) via Longstaff-Schwartz
Since $P_2$ at $T_1$ has no closed form, the **Longstaff-Schwartz algorithm** is used:

1. Simulate paths $\{S_{T_1}, S_{T_2}, S_{T_3}\}$
2. Compute $P_2|_{T_2}$ (continuation value at $T_2$)
3. Regress $P_2|_{T_2}$ onto the polynomial basis $\mathcal{B} = \{1, S_{T_1}, S_{T_1}^2, S_{T_1}^3\}$ using OLS
4. **Re-simulate** fresh paths (to avoid measurability bias)
5. Estimate $P_3$ using the approximated continuation value at $T_1$

The regression coefficients $\{\omega_k\}$ are estimated by solving:

$$A \cdot \Omega = B$$

where $A_{i,j} = \widehat{\text{Cov}}(S_{T_1}^i, S_{T_1}^j)$ and $B_i = \widehat{\text{Cov}}(P_2|_{T_2}, S_{T_1}^i)$.

### Q15 — $P_1$, $P_2$, $P_3$ as a function of $S_0$

The three prices are plotted on the same graph for $S_0 \in [0.5, 2]$:

- All three curves are **decreasing** in $S_0$ (put options)
- The hierarchy $P_1 \leq P_2 \leq P_3$ holds: more exercise dates → higher option value
- The spread between curves is larger for low $S_0$ (deep in-the-money region)

### Q16 — Quality of the Longstaff-Schwartz regression (Facultatif)
The regression quality is evaluated by comparing the LS-estimated $P_2$ against the reference Monte Carlo estimator as $N$ increases. The relative error decreases with $N$, confirming convergence of the polynomial regression approximation.

***

## Implementation Notes

### Random Number Generation
All simulations use a **uniform random generator** as the base source (as required by the problem statement), transformed via the inverse CDF method and the **Abramowitz-Stegun approximation** of the Gaussian CDF:

$$\Phi(x) \approx 1 - \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} \left( b_1 t + b_2 t^2 + b_3 t^3 + b_4 t^4 + b_5 t^5 \right), \quad t = \frac{1}{1 + b_0 x}$$

with $|\epsilon(x)| < 7.5 \times 10^{-8}$.

### Variance Reduction
Antithetic variates are used in all Monte Carlo estimations (Q9+). For each path driven by $Z \sim \mathcal{N}(0,1)$, its antithetic counterpart $-Z$ is also simulated.

### Bias-Free Regression
The Longstaff-Schwartz algorithm uses **two independent sets of trajectories**: one for the regression step and one for the valuation step, avoiding the non-measurability bias that would arise from reusing the same paths.

***

## Dependencies

```
numpy
scipy
matplotlib
```

***

## Usage

```python
from estimators import calculate_P1, Longstaff_Schwartz

# European Put
P1 = calculate_P1(r=0.02, sigma=0.2, K=1, S0=1, T=5)

# Bermudan Put (3 dates) via Longstaff-Schwartz
P3 = Longstaff_Schwartz(S0=1, r=0.02, sigma=0.2, K=1,
                         T1=1, T2=3, T3=5, N=50000, n_steps=252)
```
