"""Ergonomic checkpoint loader: point it at a checkpoint and get the fitted model in memory.

Wraps ``load_lstm_engressor_checkpoint`` with path resolution, device selection,
and a readable summary so you can load models interactively without remembering
the exact ``.pt`` path or the storage layout.

``path`` may be any of:
  - a concrete ``.pt`` file:            ``.../recurrent_lr0.003_l5_h192/checkpoint_best.pt``
  - a run directory (uses ``checkpoint_{which}.pt``):
                                        ``runs/.../lstm_stochastic_sweep_oos/recurrent_lr0.003_l5_h192``
  - a path relative to the checkpoint root (the sweep checkpoints were archived
    to ``/storage3/.../weather_checkpoints/`` mirroring the local ``runs/`` tree).

Examples
--------
```python
from experiments.load_checkpoint import load_checkpoint, load_checkpoints

# one model, best epoch, auto device (cuda if available else cpu)
eng = load_checkpoint("runs/real_data_diagnostics/ena_weather/log10_ccn/paper/"
                      "lstm_stochastic_sweep_oos/recurrent_lr0.003_l5_h192")

# several at once, keyed by config tag
models = load_checkpoints([
    ".../recurrent_lr0.003_l5_h192",
    ".../per_timestep_lr0.003_l5_h128",
    ".../global_latent_lr0.003_l3_h128",
])
eng = models["recurrent_lr0.003_l5_h192"]
samples = eng.sample(x, sample_size=400)          # use it immediately
```

From the shell (prints the summary and exits):
```bash
python experiments/load_checkpoint.py runs/.../recurrent_lr0.003_l5_h192
```
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engression_modifications.lstm import load_lstm_engressor_checkpoint
from engression_modifications.lstm.engressor import LSTMEngressor

# The sweep checkpoints were archived here, mirroring the local ``runs/`` tree.
DEFAULT_CKPT_ROOT = Path("/storage3/fs1/myu/Active/felixhu/weather_checkpoints")

# cfg flag -> human label, same names the charts use.
_VARIANT_FLAGS = (
    ("pre_additive", "LSTM + pre-additive"),
    ("stonet_head", "LSTM + StoNet"),
    ("appending_noise", "LSTM + appending noise"),
    ("additive_noise", "LSTM + additive noise"),
    ("per_timestep_noise", "LSTM + per-timestep noise"),
    ("global_latent_noise", "LSTM + global latent"),
    ("stochastic_init_noise", "LSTM + stochastic init"),
    ("recurrent_state_noise", "LSTM + recurrent noise"),
)


def _variant_label(config) -> str:
    """Map the config's noise flags to the same label used on the charts."""
    for flag, label in _VARIANT_FLAGS:
        if getattr(config, flag, False):
            return label
    return "LSTM (default head)"


def resolve_checkpoint(path: str | Path, which: str = "best", root: str | Path | None = None) -> Path:
    """Turn a ``.pt`` file / run directory / root-relative path into a concrete file.

    ``which`` selects ``checkpoint_best.pt`` or ``checkpoint_latest.pt`` when
    ``path`` is a directory. Falls back to looking under ``root`` (the archived
    checkpoint tree) when the path is not found locally.
    """
    if which not in ("best", "latest"):
        raise ValueError(f"which must be 'best' or 'latest', got {which!r}")
    p = Path(path)
    candidate = p if p.suffix == ".pt" else p / f"checkpoint_{which}.pt"
    if candidate.exists():
        return candidate
    # Fall back to the archived checkpoint root (paths mirror the local runs/ tree).
    root_path = Path(root) if root is not None else DEFAULT_CKPT_ROOT
    alt = root_path / candidate
    if alt.exists():
        return alt
    raise FileNotFoundError(
        f"no checkpoint found at {candidate} or {alt}.\n"
        f"Pass a .pt file, a run directory (uses checkpoint_{which}.pt), or a path "
        f"relative to the checkpoint root ({root_path})."
    )


def _summarize(ckpt_path: Path, eng: LSTMEngressor) -> None:
    """Print a compact, readable description of what was loaded."""
    cfg = eng.config
    input_dim = int(eng.x_mean.shape[-1])
    hidden = cfg.lstm_hidden_dim or cfg.hidden_dim
    n_params = sum(p.numel() for p in eng.model.parameters())
    print(f"loaded {_variant_label(cfg)}")
    print(f"    from      : {ckpt_path}")
    print(f"    device    : {eng.device}")
    print(f"    input_dim : {input_dim} features   |  lstm_hidden: {hidden}  "
          f"|  lstm_layers: {cfg.lstm_num_layers}")
    print(f"    head      : num_layer={cfg.num_layer}, hidden_dim={cfg.hidden_dim}, "
          f"noise_dim={cfg.noise_dim}")
    print(f"    params    : {n_params:,}")


def load_checkpoint(
    path: str | Path,
    which: str = "best",
    device: str | torch.device | None = None,
    root: str | Path | None = None,
    quiet: bool = False,
) -> LSTMEngressor:
    """Load one fitted ``LSTMEngressor`` into memory, ready to ``.sample()`` / ``.encode()``.

    Parameters
    ----------
    path:
        A ``.pt`` file, a run directory, or a path relative to ``root``.
    which:
        ``"best"`` (default) or ``"latest"`` when ``path`` is a directory.
    device:
        Compute device; defaults to cuda when available, else cpu.
    root:
        Checkpoint root for relative paths; defaults to the archived sweep tree.
    quiet:
        Suppress the summary print.
    """
    resolved = resolve_checkpoint(path, which=which, root=root)
    load_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    eng = load_lstm_engressor_checkpoint(resolved, device=load_device)
    if not quiet:
        _summarize(resolved, eng)
    return eng


def load_checkpoints(
    paths: list[str | Path],
    which: str = "best",
    device: str | torch.device | None = None,
    root: str | Path | None = None,
    quiet: bool = False,
) -> dict[str, LSTMEngressor]:
    """Load several checkpoints into a dict keyed by config tag (the run-dir name).

    The key is the config directory name (e.g. ``recurrent_lr0.003_l5_h192``),
    taken from the ``.pt`` file's parent or the directory itself.
    """
    models: dict[str, LSTMEngressor] = {}
    for path in paths:
        p = Path(path)
        tag = p.parent.name if p.suffix == ".pt" else p.name
        models[tag] = load_checkpoint(path, which=which, device=device, root=root, quiet=quiet)
    return models


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load a checkpoint and print a summary.")
    parser.add_argument("path", help=".pt file, run directory, or path relative to the checkpoint root")
    parser.add_argument("--which", choices=("best", "latest"), default="best")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--root", default=None, help="checkpoint root for relative paths")
    args = parser.parse_args()
    load_checkpoint(args.path, which=args.which, device=args.device, root=args.root)
