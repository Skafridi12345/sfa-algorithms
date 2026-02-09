# Truncated-Normal Stochastic Frontier: Likelihood Derivation

## Model Specification

The truncated-normal stochastic frontier extends the half-normal model by
allowing inefficiency to have a **non-zero mean** prior to truncation.

For a **cost frontier**, the model is:

\[
y_i = x_i^\top \beta + v_i + u_i
\]

with:
- \( v_i \sim \mathcal{N}(0, \sigma_v^2) \)
- \( u_i \sim \mathcal{N}(\mu, \sigma_u^2) \), truncated at \( u_i \ge 0 \)

This specification permits **systematic inefficiency** across units.

---

## Composite Error Distribution

Define the composite error:

\[
\varepsilon_i = v_i + u_i
\]

Since \( u_i \) is truncated, the distribution of \( \varepsilon_i \) is asymmetric
and depends on both the location parameter \( \mu \) and variance components.

Define:
\[
\sigma^2 = \sigma_v^2 + \sigma_u^2,
\quad
\lambda = \frac{\sigma_u}{\sigma_v}
\]

---

## Density of the Composite Error

The marginal density of \( \varepsilon_i \) for the truncated-normal model is:

\[
f(\varepsilon_i)
=
\frac{1}{\sigma}
\phi\!\left(
\frac{\varepsilon_i + \mu}{\sigma}
\right)
\cdot
\frac{
\Phi\!\left(
-\frac{\mu}{\sigma_u}
-
\lambda
\frac{\varepsilon_i + \mu}{\sigma}
\right)
}{
\Phi\!\left(
\frac{\mu}{\sigma_u}
\right)
}
\]

where:
- \( \phi(\cdot) \) is the standard normal density
- \( \Phi(\cdot) \) is the standard normal CDF
- The denominator ensures proper normalisation over \( u_i \ge 0 \)

---

## Log-Likelihood Function

The log-likelihood for observation \( i \) is:

\[
\ell_i(\theta)
=
-\log \sigma
+
\log \phi\!\left(
\frac{\varepsilon_i + \mu}{\sigma}
\right)
+
\log \Phi\!\left(
-\frac{\mu}{\sigma_u}
-
\lambda
\frac{\varepsilon_i + \mu}{\sigma}
\right)
-
\log \Phi\!\left(
\frac{\mu}{\sigma_u}
\right)
\]

The full log-likelihood is obtained by summation over all observations.

---

## Identification and Interpretation

Key implications of the truncated-normal specification:

- \( \mu \neq 0 \) captures **persistent inefficiency**
- Variance decomposition remains governed by \( \sigma_v \) and \( \sigma_u \)
- When \( \mu = 0 \), the model collapses to the half-normal case
- Estimation is more flexible but may suffer from weak identification
  in small or noisy samples

This model is particularly suitable when inefficiency is believed to be
**structural rather than incidental**.
