# SLURM: Regenerate the Vanilla In-Sample Chart (GPU)

One-shot GPU job that retrains the vanilla (flat) best configuration and writes its
in-sample posterior-bands chart with the corrected, model-tagged plot title
(`[vanilla (flat)]` instead of the old hardcoded `LSTM engression`). The full-data
vanilla fit OOMs / stalls on the login node, so it must run on a GPU node.

## Submit

```bash
cd /home/felixhu/WeatherEngression
sbatch slurm/regen_vanilla_chart/run.slurm
```

## What it does

1. Trains vanilla at the best-calibrated sweep config (lr 0.01, hidden 128,
   5 layers, noise 64), full paper split, `--device cuda`.
2. Writes the chart to
   `runs/real_data_diagnostics/ena_weather/log10_ccn/paper/vanilla_best/posterior_bands.png`.
3. Copies it to `reports/fig_vanilla_is.png` and rebuilds the report PDF if a TeX
   engine is on the node (otherwise run `pdflatex` on the login node afterward).

After it finishes, the report's Figure 2 vanilla panel will have the correct
in-image title. Expected coverage ~0.907 (matches the subcaption and Table 4).
