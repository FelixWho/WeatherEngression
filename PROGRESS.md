# Wildfire Effect Estimation Progress

NSF Grant motivation: marine low clouds strongly cool the planet, but aerosol effects on them are difficult to isolate because aerosols and meteorology vary together in complex ways. We study North American wildfire smoke over the remote North Atlantic and use generative models to estimate the no-wildfire distribution of CCN during wildfire periods. This counterfactual is the first step toward quantifying how wildfire aerosols alter marine clouds.

Finally, we will use these counterfactual covariate predictions to do full conditional effect estimates. So, as opposed to previous prediction/forecasting works, we are interested in causal inference.

Report: [cumulative summary](reports/cummulative_summary/summary.pdf).

## Steps

- [Synthetic data generation and testing engression quantiles](#synthetic-data-generation-and-testing-engression-quantiles) (OK)
- [EDA](#lagrangian-back-trajectory-eda) (OK)
- [Modeling CCN count from trajectories](#model-ccn-distribution-with-engression) (OK)
- [Potentially biased T-learner causal inference](#potentially-biased-t-learner) (OK)
- [Sensitivity analysis](experiments/wildfire_sensitivity/main.py) (CURRENT)
- Unbiased causal inference

Report: [cumulative summary](reports/cummulative_summary/summary.pdf).

## [Synthetic Data Generation and Testing Engression Quantiles](runs/engression_diagnostics/README.md)

We built [weather-like simulators](data_generation/synthetic_weather.py) where the true conditional law $P(Y\mid X)$ is known. The [engression diagnostic](experiments/engression_diagnostic.py) compares sampled 5th, 50th, and 95th percentiles against the true quantiles, along with 90% interval coverage.

The best in-support LSTM checks give quantile MAE 0.16 with 89% coverage for Gaussian, and 0.14 with 85% coverage for Student-t. The pre-additive check reaches 91% coverage but higher quantile MAE at 0.63. Coverage alone is not enough. These test conditional distribution learning in-support, not extrapolation.

Report: [synthetic experimental setup and results](reports/engression_experimental_setup.pdf).

## Lagrangian Back Trajectory EDA

Inspect the .mat file structure with
```
python print_mat_tables.py --mat-path data_generation/weather_data.mat
```

```
variable       kind                 shape  dtype / members
----------------------------------------------------------
CCN            double          (1, 97176)  float64
X              cell            (1, 97176)  object
period_flag    struct                      members: BB_criterion1, BB_criterion2, dust
time_ENA       double          (7, 97176)  float64
var_name       cell               (1, 22)  object

var_name (decoded feature list):
  P, LAND, PBLH^(1/5), CFLOW, CFMID, SWGDN, SLP, LWP^(1/5), WS, LTS, TS, DUST^(1/5), OMEGA, LWC^(1/5), RH, T, EPV, CO^(1/5), PREC^(1/5), CHL^(1/2), DMS^(1/2), SO2EM^(1/5)
```

### Data Basics 

Ingest .mat data into Python objects with the [ENA loader](data_generation/ena_weather.py):
```
python data_generation/ena_weather.py
```

```
======================================================================
1. LOAD
======================================================================
  mat_path : data_generation/weather_data.mat
  target   : ccn   log_ccn=False   max_samples=all   seq_stride=4

...

======================================================================
9. PREVIEW (first sample, first 5 of 61 steps)
======================================================================
  source #13276  y=101.1
[[   894.4        0.         3.24       0.515      0.         0.    100814.586      0.425      8.324     22.829    263.865      0.013
      -0.15       0.075      0.642    267.342      0.         0.042      0.         0.         0.342      0.001]
 [   881.8        0.         3.097      0.867      0.123      0.    100748.99       0.581      8.641     23.784    263.481      0.013
      -0.03       0.035      0.577    266.609      0.         0.042      0.         0.209      0.275      0.001]
 [   874.4        0.         2.566      0.826      0.454     82.656 100883.734      0.497      8.647     23.681    263.037      0.013
      -0.01       0.026      0.41     266.915      0.         0.042      0.         0.296      0.307      0.001]
 [   862.1        0.         3.094      0.846      0.299    484.75  100807.7        0.488      8.377     21.481    265.359      0.013
      -0.17       0.017      0.437    266.5        0.         0.042      0.         0.296      0.403      0.002]
 [   810.2        1.         3.354      0.812      0.403    198.688 100757.5        0.578      9.434     15.599    267.435      0.013
      -0.142      0.001      0.469    261.442      0.         0.041      0.         0.         0.         0.003]]

======================================================================
SUMMARY
======================================================================
  ALL CHECKS PASSED
```

Each observation is a 10-day airmass back-trajectory: 241 hourly timesteps by 22 weather covariates, with $\log_{10}(\mathrm{CCN})$ at arrival as the target. We use the [paper split](resources/weather.pdf), which holds out January, March, May, July, September, and November of 2022. However, we do not use the paper's validation methods.

One consideration is the rows are heavily autocorrelated. Adjacent hourly trajectories overlap heavily.

### [Correlations](experiments/correlation_heatmap.py)

No single covariate explains CCN. Using each covariate's mean over the trajectory, the largest marginal correlation is downwelling solar radiation at about $+0.33$. Cloud water, low-cloud fraction, and wind speed are next at about $-0.25$. Pearson and Spearman correlations are similar, suggesting that most of the nonlinearity is in how the covariates interact rather than in any one marginal relationship.

### [In-sample vs Out-of-sample Covariate Distribution Shift](reports/distribution_shift_testing/summary.pdf)

It's difficult to understand what it means for an out-of-sample $X$ to be from a shifted distribution relative to in-sample $X$. We try a couple methods.

The [train/test shift diagnostics](experiments/distribution_shift_testing/run.py) find that held-out weather is different from the training weather, mainly through CO and air-mass-source variables. A classifier separates train from test with AUC 0.79, and MMD confirms a real shift ($p=0.0008$). Still, test trajectories mostly remain inside training support: the median nearest-neighbor distance ratio is 1.04, with 7.4% of test episodes beyond the training 95th percentile.

[The naive wildfire-versus-clean comparison](experiments/covariate_wildfire_shift.py) is badly confounded by season because 84% of wildfire trajectories occur from June through September. After matching the clean group by calendar month, CO has the largest remaining shift, followed by LAND, PBLH, and LWP. This is descriptive, not yet causal: LAND, for example, is not changed by smoke; it indicates the continental paths that collect smoke. We therefore cannot label a covariate wildfire-sensitive from EDA alone.

Reports: [distribution-shift tests](reports/distribution_shift_testing/summary.pdf) and [real-data summary](reports/real_data_engression_summary.pdf).

## Model CCN Distribution With Engression

The [real-data diagnostic](experiments/real_data_diagnostic.py) tries a variety of models. Some of the better performers include [global-latent, per-timestep, and recurrent](engression_modifications/lstm/generators.py).

In global-latent, our best-performing model, we concatenate a universal epsilon at every timestep. Suppose epsilon is 96-dimensional. The data, originally of shape (241, 22), becomes (241, 118). In per-timestep, a fresh epsilon is appended on every timestep.

We evaluate the models visually and statistically. Since all engression models $F$ are generative. The probability the true $y$ lies at or below the value $F(X)$ should be uniform due to inverse transform sampling. We [visually inspected](experiments/pit_uniformity.py) and found global-latent to be most uniform-like.

Then we use Anderson-Darling test for uniformity to confirm our observations.

$$
H_0: u_i \sim \mathrm{Uniform}(0,1),
\qquad
H_1: u_i \not\sim \mathrm{Uniform}(0,1),
$$

where $u_{(i)}$ is the $i$ th sorted PIT value:

$$
A^2=-n-\frac{1}{n}\sum_{i=1}^{n}(2i-1)
\left[\log u_{(i)}+\log\left(1-u_{(n+1-i)}\right)\right].
$$

Approximate critical values are 1.93 at 10%, 2.49 at 5%, 3.07 at 2.5%, and 3.86 at 1%. Reject $H_0$ when $A^2$ exceeds the chosen value; our test computes the finite-sample value by Monte Carlo.

Report: [real-data engression results](reports/real_data_engression_summary.pdf).

## [Potentially Biased T-learner](experiments/t_learner/train.py)

Evaluation: [T-learner test](experiments/t_learner/test.py).

Report: [T-learner effect summary](reports/t_learner_effect/summary.pdf).

## Sensitivity Analysis

We start with a basis of insensitive and sensitive variables and iteratively see how many we can move from the sensisitive into the insensitive section. [Code](experiments/wildfire_sensitivity/main.py).

Our naive starting basis is
```
[
        "P",
        "LAND",
        "SLP",
        "WS",
        "TS",
        "DUST^(1/5)",
        "EPV",
        "CHL^(1/2)",
        "DMS^(1/2)",
        "SO2EM^(1/5)",
    ]
```

For a single sweep and sweep-until-stable along the sensitives, we get a final insensitive list:

```
['P', 'LAND', 'SLP', 'WS', 'TS', 'DUST^(1/5)', 'EPV', 'CHL^(1/2)', 'DMS^(1/2)', 'SO2EM^(1/5)', 'CFLOW', 'CFMID', 'LTS', 'OMEGA', 'LWC^(1/5)', 'RH', 'T']
```

The concern I have is whether the list itself is sensitive to the order in which we process sensisitive variables. Additionally, I need to write some code to ensure the counterfactual model is calibrated.

## Unbiased/Adjusted Causal Inference

Final models:

1. Model 1:  M  →  M' (predicted)
2. Model 2:  M + M' (predicted)  →  CCN
3. Model 3:  M + observed wildfire-period M' + wildfire emissions  →  CCN

Model 1 is our counterfactual model. Model's 2 and 3 predict what would happen with/without wildfire plumes.

Steps:

1. Model 1 is trained only on clean periods. During a wildfire period, feed it the observed robust variables. It generates the sensitive meteorology that would plausibly have occurred without wildfire: M'_0.
2. Model 2 is also trained only on clean periods. Feed it the robust variables and generated M'_0. It generates the counterfactual no-wildfire CCN distribution: CCN_0.
3. Model 3 is trained primarily on wildfire periods, with clean observations added for stability. It generates the factual wildfire-period CCN distribution: CCN_1, conditioned on robust variables, observed sensitive variables, and wildfire emissions.