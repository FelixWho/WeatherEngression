# OOS Comparison: narx_student_t / regularized / d=12

This run fits one model on one shared training set, then evaluates it on
held-out subsets chosen by different out-of-support definitions.

## Command

```bash
python experiments/oos_comparison.py --engression-model regularized --lr 0.003 --weight-decay 0.003
```

## Outputs

- `results.csv`: summary metrics by OOS definition.
- `results.json`: run metadata and the same metrics.
