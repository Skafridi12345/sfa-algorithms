# Estimation Algorithm: Half-Normal SFA
## > Theory-only exposition. No executable code or proprietary implementation included.


## Objective

Estimate parameters:

\[
\theta = (\beta, \sigma_v, \sigma_u)
\]

by **maximum likelihood**, accounting for the asymmetric composite error structure.

---

## Algorithm Outline

**Step 1 — Initialisation**
- Estimate \( \beta^{(0)} \) using Ordinary Least Squares
- Compute residual variance
- Initialise \( \sigma_v \) and \( \sigma_u \) as fractions of residual scale

**Step 2 — Likelihood Construction**
- Form the log-likelihood function:

\[
\ell(\theta) =
\sum_i
\left[
-\log(\sigma)
+ \log \phi\left(\frac{\varepsilon_i}{\sigma}\right)
+ \log 2
+ \log \Phi\left(
\frac{\lambda \varepsilon_i}{\sigma}
\right)
\right]
\]

where:
- \( \phi(\cdot) \) is the standard normal density
- \( \Phi(\cdot) \) is the standard normal CDF

**Step 3 — Optimisation**
- Maximise \( \ell(\theta) \) using a quasi-Newton method (e.g. BFGS / L-BFGS)
- Enforce positivity of variance parameters via log-transform

**Step 4 — Convergence Checks**
- Verify gradient norm
- Inspect Hessian conditioning
- Confirm reasonable parameter magnitudes

**Step 5 — Post-Estimation**
- Recover structural parameters
- Compute inefficiency estimates
- Derive technical efficiency scores

---

## Numerical Considerations

- Likelihood surface may be flat when \( \sigma_u \approx 0 \)
- Poor scaling can cause optimisation failure
- Analytical gradients significantly improve stability


## Algorithm: Half-Normal Cost Frontier (MLE)

Input: y, X
Initialise β via OLS
Initialise σ_v, σ_u from residual variance

Repeat until convergence:
    Compute ε = y − Xβ
    Evaluate log-likelihood ℓ(β, σ_v, σ_u)
    Update parameters via quasi-Newton step

Output: β̂, σ̂_v, σ̂_u

### Theory-only exposition. No executable code or proprietary implementation included.

