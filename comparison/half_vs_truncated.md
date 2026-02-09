# Half-Normal vs Truncated-Normal SFA
## > Theory-only exposition. No executable code or proprietary implementation included.


## Key Conceptual Difference

| Aspect | Half-Normal | Truncated-Normal |
|------|------------|------------------|
| Mean of inefficiency | Zero | Non-zero |
| Flexibility | Low | Higher |
| Identification risk | Low | Moderate |
| Use case | Baseline | Structural inefficiency |

---

## Half-Normal Model
- Assumes inefficiency is centered at zero
- Suitable when inefficiency is expected to be minimal or random
- Often used as a **benchmark specification**

## Truncated-Normal Model
- Allows inefficiency to have a non-zero mean
- Captures **systematic inefficiency**
- More flexible but harder to estimate
- Risk of weak identification in small samples

---

## Practical Guidance

- Start with half-normal as a baseline
- Use truncated-normal as a robustness or structural extension
- Compare likelihoods and efficiency distributions
- Inspect sensitivity of efficiency rankings

---

## Interpretation for Benchmarking

In applied efficiency analysis (e.g. energy, utilities, buildings):
- Half-normal → conservative efficiency estimates
- Truncated-normal → captures persistent underperformance
