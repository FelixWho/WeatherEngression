# Experiments

This package contains reusable experiment infrastructure for synthetic
weather-engression work.

## Main Diagnostic

Run a vanilla engression diagnostic:

```bash
python experiments/engression_diagnostic.py
```

Run the classic Adam regularized variant:

```bash
python experiments/engression_diagnostic.py \
  --engression-model regularized \
  --weight-decay 0.003
```

Run the sequence-native LSTM engression variant:

```bash
python experiments/engression_diagnostic.py \
  --engression-model lstm \
  --model narx_student_t \
  -d 12
```

By default, outputs are written under:

```text
runs/engression_diagnostics/<data-model>/<split>/<engression-model>/
```

## Central Pipeline

For reusable code, prefer the central pipeline instead of hand-wiring data
generation, splitting, fitting, prediction, and metrics:

```python
from experiments import (
    EngressionFitConfig,
    OOSConfig,
    PredictionConfig,
    SplitConfig,
    SyntheticDataConfig,
    run_engression_experiment,
)

result = run_engression_experiment(
    data_config=SyntheticDataConfig(data_model="narx_student_t", x_dimension=12),
    split_config=SplitConfig(split="in-support", train_size=4000, test_size=500),
    fit_config=EngressionFitConfig(
        engression_model="regularized",
        lr=0.003,
        weight_decay=0.003,
        num_epochs=120,
    ),
    prediction_config=PredictionConfig(sample_size=800),
    oos_config=OOSConfig(knn_threshold_quantile=0.95),
)

print(result.metrics)
```

This is the main exchange point:

\[
\text{synthetic data config}
\rightarrow
\text{split config}
\rightarrow
\text{engression model config}
\rightarrow
\text{OOS diagnostics}
\rightarrow
\text{metrics and artifacts}.
\]

MLP-style variants receive flattened lag windows with shape
\((n, (L+1)d)\). The LSTM variant receives sequence inputs with shape
\((n, L+1, d)\). OOS diagnostics still use flattened \(X\) for all models so
support definitions are comparable.

## Out-Of-Support Diagnostics

The pipeline computes four OOS diagnostics for every held-out row:

1. `scalar_projection`: outside the training range of \(\phi(X)\).
2. `marginal_range`: at least one flattened \(X\) feature lies outside the
   training min/max range.
3. `marginal_quantile`: at least one flattened \(X\) feature lies outside the
   training central quantile range, defaulting to
   \([Q_{0.01}, Q_{0.99}]\).
4. `knn_distance`: standardized nearest-neighbor distance to training \(X\)
   exceeds a train-to-train kNN distance threshold.

Summaries are stored in:

```python
result.metrics["oos"]
```

Row-level boolean labels are stored in:

```python
result.oos.flags
```

To compare the same fitted vanilla engression model across all OOS definitions:

```bash
python experiments/oos_comparison.py
```

The comparison runner uses one shared training set, one fitted model, and then
reports summary metrics on held-out subsets selected by each OOS definition.
By default, outputs are written under:

```text
runs/oos_comparisons/<data-model>_<engression-model>_d<dimension>/
```

## Modules

- `constants.py`: shared quantile keys, split names, and display labels.
- `pipeline.py`: central generated-data experiment pipeline.
- `oos.py`: scalar, marginal, quantile-marginal, and kNN OOS diagnostics.
- `splits.py`: train/test splits, including \(\phi(X)\)-based extrapolation.
- `data.py`: dataset-to-tensor conversion for flattened lag-window inputs.
- `predictions.py`: sample-based engression quantile prediction.
- `metrics.py`: quantile error, interval coverage, and interval width metrics.
- `plotting.py`: posterior-band chart generation.
- `artifacts.py`: run README and metrics artifact writing.
- `engression_diagnostic.py`: command-line diagnostic runner.
- `oos_comparison.py`: one-model comparison across OOS definitions.

## Design Boundary

Synthetic data generation stays in `generate_data.py` and `data_generation/`.
Engression model variants stay in `engression_modifications/`. This package is
the glue layer that runs experiments using those pieces.
