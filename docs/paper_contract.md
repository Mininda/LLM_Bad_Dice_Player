# Reproducibility Notes

This document records the canonical settings used by the cleaned release repository.

## Main benchmark

- 15 distributions
- 11 models
- sample count: `N=1000`
- default decoding: `temperature=1.0`, `top_p=1.0`
- continuous validity: two-sample KS against the released reference samples
- discrete validity: chi-square goodness-of-fit against the theoretical PMF
- significance threshold: `alpha=0.01`

Canonical parameters are defined in `configs/distributions.json` and match the final paper:
- Uniform: `U(0, 1)`
- Gaussian: `N(0, 1)`
- Bernoulli: `p = 0.7`
- Beta: `alpha = 2, beta = 2`
- Binomial: `n = 10, p = 0.5`
- Poisson: `lambda = 5`
- Exponential: `lambda = 1`
- Cauchy: `x0 = 0, gamma = 1`
- Student's `t`: `df = 3`
- Chi-Square: `df = 5`
- F-Distribution: `d1 = 5, d2 = 10`
- Gamma: `shape = 2, scale = 2`
- Weibull: `shape = 1.5, scale = 1`
- Laplace: `loc = 0, scale = 1`
- Logistic: `loc = 0, scale = 1`

## Downstream tasks

### MCQ
- `N=1000`
- temperature `1.0`
- target distribution: uniform answer position over `A/B/C/D`
- test: chi-square goodness-of-fit

### Joint attributes
- `N=1000`
- temperature `1.0`
- gender target: `49.49 / 50.51`
- race target: `57.46 / 20.02 / 12.63 / 6.49 / 3.40`
- height target: `N(169.0, 10.0^2)`
- coat color target: uniform over 7 colors
- categorical tests: chi-square goodness-of-fit
- height test: one-sample KS against `N(169, 10^2)`

### Independent attribute follow-ups
- `N=1000`
- temperature `1.0`
- same targets as the joint task
- categorical tests: chi-square goodness-of-fit
- height test: one-sample KS against `N(169, 10^2)`

## Prompt files

- Main benchmark prompts: `prompts/batch/` and `prompts/independent/`
- Downstream prompts: `prompts/downstream/`

The cleaned runners in `src/generate_samples.py` and `src/generate_downstream.py` use these prompt files directly.
