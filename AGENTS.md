# Agent Context

This repo is dedicated to a research project on engression for weather-like time-series prediction.

## Source Papers

The relevant papers are in `resources/`:

- `resources/engression.pdf`: introduces engression, a neural distributional regression method trained with energy loss. Important ideas: conditional sampler $g_\theta(X,\varepsilon)$, energy score, pre-additive noise, extrapolation under monotonicity/regularity assumptions.
- `resources/weather.pdf`: proposes a Lagrangian time-series ML framework for predicting cloud condensation nuclei from airmass trajectory histories. Important ideas: each sample is a multivariate time series along an airmass trajectory; the target is aerosol concentration at arrival.

## Current Research Framing

The project should be framed as:

> Probabilistic Lagrangian / weather-like forecasting with engression-style distributional regression.

Avoid overstating the extrapolation theorem. The safest claim is that engression can learn calibrated predictive conditional distributions $P(Y \mid X=x)$. Strong extrapolation claims require the synthetic or real data-generating process to satisfy additional pre-additive and monotonicity assumptions.

## Mental Model To Preserve

In pre-additive noise,

$$
Y=g(X+\eta).
$$

The observed $x$ is not itself corrupted. Instead, $x$ is the anchor, $\eta$ is a latent horizontal perturbation, and applying $g$ turns horizontal uncertainty into vertical conditional spread:

$$
x+\eta \mapsto g(x+\eta) \sim Y \mid X=x.
$$

Engression uses artificial base noise $\varepsilon$, then learns or biases toward a transformation

$$
\eta=h_\theta(\varepsilon).
$$

## Repo Organization

- `data_generation/`: synthetic weather-like time-series simulators with known conditional laws.
- `engression_modifications/`: the engression models. All four report architectures live here:
  vanilla flat (`vanilla_engression.py`) and the LSTM encoder with default / StoNet / pre-additive
  heads (the `lstm/` package, selected by `--stonet-head` / `--pre-additive`).
- `archive/`: superseded prototype code, imported by nothing (see `archive/README.md`).
- `resources/`: PDFs only; do not edit or overwrite paper PDFs.

## Storage

Large data lives outside the repo. The main storage directory for this project is:

```text
/storage3/fs1/myu/Active
```

The real Eastern North Atlantic (ENA) weather dataset is a MATLAB v7.3 (HDF5) file at:

```text
/storage3/fs1/myu/Active/felixhu/weather_data.mat
```

Read it with `h5py` (the file is HDF5 under the hood). Each sample is an airmass back-trajectory of weather variables with a scalar cloud-condensation-nuclei (CCN) target.

## Running Jobs (IMPORTANT)

The login node has no GPU and limited/contended CPU. Do **not** run training,
evaluation, sweeps, or any heavy/long-running compute in the foreground with
`.venv/bin/python experiments/....py`. Foreground runs hog the login node's CPU,
load the ~3 GB `.mat` slowly over the network filesystem, and get killed before
finishing.

Instead:

- **Always submit heavy work as a SLURM batch job.** Write (or reuse) a `.slurm`
  file under `slurm/<task>/run.slurm` and `sbatch` it. Follow the existing files
  in `slurm/` as templates (partition `general-gpu`, `source .venv/bin/activate`,
  a CUDA sanity check, then the `python -u ...` call).
- **Always charge the `compute2-myu` account.** Every `.slurm` file must carry

  ```bash
  #SBATCH -A compute2-myu
  ```

  This applies to every job from here on, not just the existing ones. Omitting
  `-A` silently falls back to the submitting user's default account, so the run
  is not charged to this project's allocation. All 41 current files in `slurm/`
  already use it; match them.
- **Use the GPU whenever possible:** `-p general-gpu`, `#SBATCH -G 1`, and pass
  `--device cuda` to the script. Training that OOMs or crawls on the login node
  runs fine on a GPU node.
- Only tiny, seconds-long sanity checks (imports, argument parsing, a shape print)
  may be run directly; anything that loads the full dataset or trains a model goes
  through SLURM.
- If a `.slurm` for the task does not exist yet, create one rather than running the
  command directly.
- **Always save checkpoints.** Any training job must write model checkpoints (for
  the LSTM diagnostic that means leaving checkpointing ON — do NOT pass
  `--no-checkpoint`). A trained model with no checkpoint cannot be reloaded, so
  post-hoc work (OOS diagnostics, OOD splits, chart regeneration, retraining a
  winner) forces a full re-run. The extra disk is far cheaper than the wasted GPU
  hours. This applies to sweeps too: prefer keeping checkpoints even for lean
  grid runs.

## Synthetic Models

Synthetic weather types are split into one module per target law:

- `data_generation/preadditive.py`: the theory-friendly baseline $Y=g(\phi(X)+\eta)$.
- `data_generation/narx_gaussian.py`: nonlinear heteroskedastic Gaussian $Y\mid X=x$.
- `data_generation/narx_student_t.py`: nonlinear heteroskedastic Student-t target with heavy-tailed shocks.
- `data_generation/narx_garch.py`: nonlinear NARX mean with GARCH-style volatility clustering. Its saved one-step conditional law is $Y_t\mid X_t,\mathcal{F}_{t-1}$ because the variance remembers previous generated residuals.
- `data_generation/regime_mixture.py`: X-dependent mixture of Gaussian weather regimes.
- `data_generation/hurdle_lognormal.py`: zero-inflated precipitation-like target.

`data_generation/common.py` owns shared covariate simulation and lag-window utilities. `data_generation/synthetic_weather.py` should stay thin: CLI, model dispatch, and true-law reference sampling.

Use the non-preadditive models to test robustness when the engression paper's structural assumptions are violated but the true conditional law is still known.

## Communication Style

- Do not be verbose. Concise and intuitive is the priority.
- Lead with the answer; cut preamble, hedging, and exhaustive option lists.
- Favor plain intuition over formal walls of text; add detail only when asked.
- Match the answer's length to the question. A yes/no question gets yes or no
  first, then a sentence or two of why. Not a restructured essay.
- Do not add structure to short answers. No bolded headings, no bullet lists,
  no tables for anything under roughly a paragraph. Just say it.
- Do not volunteer caveats, edge cases, sizing wrinkles, or next steps that were
  not asked for. If a caveat genuinely matters, one sentence, at the end.
- Write like a normal person talking. Plain verbs, ordinary words. No flourish,
  no metaphor, no rhetorical shape.
- Banned phrasings, as examples of the style to avoid: "it lands", "that tracks",
  "the case that proves your point", "worth your attention", "here is the thing",
  and any sentence fragment used for emphasis. Say "you're right", "yes", "this is
  a problem because X". Agreement and disagreement get stated, not dramatized.
- Do not editorialize findings with words like sharp, troubling, striking, telling,
  or damning. State the fact and let it stand.

## Implementation Principles

- For wildfire-sensitivity analysis, use only results from `until_stable`
  experiments in summaries, comparisons, and reports. Single-pass runs are not
  part of the reported evidence going forward.

- Keep synthetic data generators explicit enough that the true conditional distribution is available analytically or by a known sampler.
- Prefer simple, inspectable data-generating processes before adding realism.
- Keep model code separate from data generation.
- Use LaTeX math in documentation.
- For validation, prioritize calibration and distributional metrics over only RMSE:
  - quantile calibration;
  - prediction interval coverage;
  - CRPS / energy score;
  - tail-bin performance;
  - generated sample diagnostics.
