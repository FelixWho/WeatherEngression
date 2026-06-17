# Runs

This folder is for generated experiment outputs. Each run should live in its
own subfolder with a local `README.md` describing the command, parameters,
outputs, and interpretation.

## Layout

```text
runs/
├── synthetic_data/
│   └── <model-or-gallery>/
│       ├── README.md
│       └── target_timeseries.png
├── engression_diagnostics/
    └── <model>/
        └── <split>/
            ├── README.md
            ├── metrics.json
            └── posterior_bands.png
└── optimizer_sweeps/
    └── classic_adam/
        └── <model>_<split>_d<d>/
            ├── README.md
            ├── results.csv
            └── results.json
```

## Synthetic Data Runs

Synthetic visualization runs are created by:

```bash
python plot_synthetic_data.py --model narx_student_t
```

By default, this writes to:

```text
runs/synthetic_data/narx_student_t/
```

The folder README explains the plotted target \(Y_t\), the lag-window input
\(X_t=(W_{t-L},\ldots,W_t)\), and the generation parameters.

## Engression Diagnostic Runs

Engression posterior-band diagnostics are created by:

```bash
python experiments/engression_diagnostic.py --model narx_student_t --split in-support
```

By default, this writes to:

```text
runs/engression_diagnostics/narx_student_t/in-support/vanilla/
```

The folder README explains the split, the model settings, and the main
calibration metrics. The `metrics.json` file stores the same scalar results in
machine-readable form.

## Optimizer Sweeps

Classic Adam optimizer sweeps are created by:

```bash
python engression_modifications/classic_adam_sweep.py
```

By default, this writes to:

```text
runs/optimizer_sweeps/classic_adam/narx_student_t_in-support_d12/
```

The folder README records the best setting by a heuristic calibration score,
and `results.csv` contains the full sweep table.

## Notes

The run folders are generated artifacts. Re-running a command with the same
model and split will overwrite the matching plot, README, and metrics file.
