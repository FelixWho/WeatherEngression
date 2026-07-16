# SLURM: Regenerate the Default-Head Charts (GPU)

One-shot GPU job that regenerates the **LSTM + default head** best run (the 0.888
config that was overwritten when the StoNet run reused the `.../lstm/` folder). It
produces both the in-sample posterior-bands chart and the out-of-sample
coverage-vs-distance chart (the LSTM encoder provides an embedding, so OOS applies),
correctly titled by the fixed plotting code, and drops them into the report's
`\IfFileExists` slots.

## Submit

```bash
cd /home/felixhu/WeatherEngression
sbatch slurm/regen_default_charts/run.slurm
```

## What it does

1. Trains the default head (LSTM, no `--stonet-head`/`--pre-additive`; lr 0.003,
   hidden 192, 3 layers, noise 96), full paper split, `--device cuda`, OOS on.
2. Writes to `runs/real_data_diagnostics/ena_weather/log10_ccn/paper/default_head/`
   (distinct folder, does not clobber the StoNet run).
3. Copies `posterior_bands.png` -> `reports/fig_default_is.png` and
   `coverage_vs_distance.png` -> `reports/fig_default_oos.png`, then rebuilds the PDF.

After it runs, Figures 2 and 3 in the report auto-fill the default-head panels
(their `\IfFileExists` placeholders become the real charts on recompile).

## Note

The flat **vanilla** model has no learned embedding, so its OOS panel is a
deliberate "N/A" (raw 5302-d distance is degenerate). Only the three LSTM-based
heads get an embedding-space OOS chart. The vanilla in-sample chart is regenerated
separately by `slurm/regen_vanilla_chart/`.
