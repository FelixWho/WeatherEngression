# OOS Comparison: narx_student_t / lstm / d=12

This run fits one model on one shared training set, then evaluates it on
held-out subsets chosen by different out-of-support definitions.

## Command

```bash
python experiments/oos_comparison.py --engression-model lstm --model narx_student_t -d 12 --num-samples 5000 --train-size 4000 --max-test-size 500 --window 12 --epochs 40 --batch-size 512 --hidden-dim 192 --noise-dim 96 --lr 0.003 --weight-decay 0 --prediction-samples 800 --out-dir runs/oos_comparisons/narx_student_t_lstm_e40_wd0_d12
```

## Outputs

- `results.csv`: summary metrics by OOS definition.
- `results.json`: run metadata and the same metrics.
