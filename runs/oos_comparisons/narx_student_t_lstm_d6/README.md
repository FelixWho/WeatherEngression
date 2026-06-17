# OOS Comparison: narx_student_t / lstm / d=6

This run fits one model on one shared training set, then evaluates it on
held-out subsets chosen by different out-of-support definitions.

## Command

```bash
python experiments/oos_comparison.py --engression-model lstm --model narx_student_t -d 6 --num-samples 180 --train-size 120 --max-test-size 20 --window 4 --epochs 1 --batch-size 32 --hidden-dim 16 --noise-dim 8 --prediction-samples 20 --oos-knn-reference-size 80 --oos-knn-batch-size 32
```

## Outputs

- `results.csv`: summary metrics by OOS definition.
- `results.json`: run metadata and the same metrics.
