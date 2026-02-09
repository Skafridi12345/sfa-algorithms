# Half-Normal Stochastic Frontier Model (Cost Frontier)

## Model Definition

The half-normal stochastic frontier model decomposes deviations from the
frontier into **random noise** and **one-sided inefficiency**.

For a **cost frontier**, the model is specified as:

\[
y_i = x_i^\top \beta + v_i + u_i
\]

where:

- \( y_i \) : observed cost (or energy consumption)
- \( x_i \) : vector of explanatory variables
- \( \beta \) : technology parameters
- \( v_i \sim \mathcal{N}(0, \sigma_v^2) \) : symmetric noise
- \( u_i \ge 0 \) : inefficiency term

## Inefficiency Distribution

The inefficiency component is assumed to follow a **half-normal distribution**:

\[
u_i \sim |\mathcal{N}(0, \sigma_u^2)|
\]

This implies:
- Inefficiency has zero mode
- All deviations are non-negative
- The model represents a parsimonious baseline specification

## Cost Frontier Interpretation

Because inefficiency enters **additively and positively**, larger values of \( u_i \)
correspond to **higher observed costs**, conditional on inputs.

This formulation is appropriate for:
- Energy benchmarking
- Resource overuse analysis
- Operational inefficiency measurement

## Composite Error Structure

Define the composite error:

\[
\varepsilon_i = v_i + u_i
\]

The density of \( \varepsilon_i \) is asymmetric and depends on:

\[
\sigma^2 = \sigma_v^2 + \sigma_u^2,
\quad
\lambda = \frac{\sigma_u}{\sigma_v}
\]

These parameters control the relative contribution of noise versus inefficiency.
