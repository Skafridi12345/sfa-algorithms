# Estimation Algorithm: Truncated-Normal SFA

## Parameter Vector

\[
\theta = (\beta, \mu, \sigma_v, \sigma_u)
\]

where \( \mu \) governs the mean of inefficiency prior to truncation.

---

## Algorithm Outline

**Step 1 — Initialisation**
- Estimate \( \beta^{(0)} \) using OLS
- Initialise \( \mu^{(0)} = 0 \)
- Initialise variance parameters from residual scale

**Step 2 — Likelihood Construction**
- Incorporate truncation normalisation term
- Ensure numerical stability of CDF evaluations
- Apply log-transforms to enforce variance positivity

**Step 3 — Optimisation**
- Maximise log-likelihood using quasi-Newton methods
- Prefer analytical gradients where available
- Monitor step size and curvature

**Step 4 — Diagnostics**
- Check convergence robustness
- Inspect correlation between \( \mu \) and \( \sigma_u \)
- Compare likelihood with half-normal benchmark

**Step 5 — Model Validation**
- Assess efficiency distribution shape
- Perform likelihood ratio testing against half-normal model
- Evaluate sensitivity of efficiency rankings

---

## Practical Considerations

- Strong collinearity may impair identification
- Small samples can lead to unstable \( \mu \) estimates
- Use as an extension rather than a default specification
