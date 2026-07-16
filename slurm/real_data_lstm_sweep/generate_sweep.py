"""Generate one SLURM script per (parameter-set x head) for the real-data sweep.

Each run trains the LSTM engression model on the ENA log10(CCN) / paper split and
writes to its own --out-dir so runs never clobber each other. Two heads are swept:

- stonet : loose package StoNet head (noise at input + every layer, non-monotone).
- preadd : engression-paper pre-ANM head Y = g(phi(X) + eta) with monotone g.

Re-run this file to regenerate the .slurm scripts after editing the grid.
"""

from __future__ import annotations

from pathlib import Path

SWEEP_DIR = Path(__file__).resolve().parent
OUT_ROOT = "runs/real_data_diagnostics/log10_ccn/paper/sweep"

# Parameter sets probing the diagnosed failure (model barely conditions on X):
# shorter sequences, more capacity, regularization, lower LR.
PARAM_SETS = [
    dict(tag="A_full",   seq_stride=1, hidden=192, noise=96,  num_layer=3, lr=0.003, wd=0.0,    epochs=120, time="08:00:00"),
    dict(tag="B_short",  seq_stride=4, hidden=192, noise=96,  num_layer=3, lr=0.003, wd=0.0,    epochs=120, time="04:00:00"),
    dict(tag="C_bigreg", seq_stride=4, hidden=256, noise=128, num_layer=3, lr=0.003, wd=1e-4,   epochs=150, time="05:00:00"),
    dict(tag="D_lowlr",  seq_stride=4, hidden=192, noise=96,  num_layer=3, lr=0.001, wd=1e-4,   epochs=200, time="06:00:00"),
]

HEADS = ("stonet", "preadd")

TEMPLATE = """#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-sw-{tag}-{head}
#SBATCH --output=slurm/real_data_lstm_sweep/{tag}_{head}_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={time}

set -euo pipefail

cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

# Sweep cell: param-set {tag}, head '{head}'.
python -u experiments/real_data_diagnostic.py \\
  --target ccn \\
  --log-ccn \\
  --split paper \\
  --max-samples all \\
  --train-size all \\
  --test-size all \\
  --seq-stride {seq_stride} \\
  --epochs {epochs} \\
  --batch-size 256 \\
  --hidden-dim {hidden} \\
  --noise-dim {noise} \\
  --num-layer {num_layer} \\
  {head_flags} \\
  --lr {lr} \\
  --weight-decay {wd} \\
  --prediction-samples 800 \\
  --checkpoint-every 10 \\
  --device cuda \\
  --skip-assertions \\
  --out-dir {out_dir}
"""


def head_flags(head: str, noise: int) -> str:
    if head == "stonet":
        return "--stonet-head"
    if head == "preadd":
        return f"--pre-additive \\\n  --index-dim {noise}"
    raise ValueError(head)


def main() -> None:
    written = []
    for ps in PARAM_SETS:
        for head in HEADS:
            out_dir = f"{OUT_ROOT}/{ps['tag']}_{head}"
            script = TEMPLATE.format(
                head_flags=head_flags(head, ps["noise"]),
                out_dir=out_dir,
                head=head,
                **ps,
            )
            path = SWEEP_DIR / f"run_{ps['tag']}_{head}.slurm"
            path.write_text(script, encoding="utf-8")
            written.append(path.name)

    submit = ["#!/bin/bash", "# Submit every sweep cell.", "set -euo pipefail", ""]
    submit += [f"sbatch slurm/real_data_lstm_sweep/{name}" for name in written]
    (SWEEP_DIR / "submit_all.sh").write_text("\n".join(submit) + "\n", encoding="utf-8")

    print(f"wrote {len(written)} slurm scripts + submit_all.sh to {SWEEP_DIR}")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
