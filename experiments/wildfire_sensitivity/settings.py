"""Fixed sensitivity defaults: seeds, split fractions, sampling, and decisions.

CLI overrides remain in main.build_parser(). Changing values here changes the
experiment; the split-feasibility diagnostics do not override these defaults.
"""
import torch

from experiments.t_learner.train import DEFAULT_CKPT_ROOT


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SWEEP_SEED = 2026
PREDICTION_SAMPLES = 400
# The PIT sweep used 256.  MPS retains LSTM backward activations in unified
# memory, so its 20 GB limit cannot accommodate this global-latent model at
# that batch size.
TRAINING_BATCH_SIZE = 64 if DEVICE == "mps" else 256
# Sampling all 6,786 wildfire trajectories at once creates 400 full
# conditional trajectory draws in MPS memory.  Evaluate in small batches.
EVALUATION_BATCH_SIZE = 64 if DEVICE == "mps" else 1_024

SEQ_STRIDE = 1
# Fraction of clean arrival-time blocks held back as the control group.
HOLDOUT_FRACTION = 0.25
HOLDOUT_BLOCK_DAYS = 30.0
# Fraction of the remaining clean training blocks held back for early stopping.
# Separate from the control and from the rows used to fit quantile corrections.
VALIDATION_FRACTION = 0.15
# Separate clean rows for fitting the pooled quantile correction.
RECALIBRATION_FRACTION = 0.20
# A candidate counts as sensitive when its wildfire coverage falls this far
# below the clean control's. The margin is a judgment call rather than a
# calibrated significance test; uncertainty in the coverage difference is not tested.
SENSITIVITY_MARGIN = 0.02

# Central-interval levels reported by print_calibration, and PIT histogram bins.
CALIBRATION_LEVELS = (0.50, 0.75, 0.90, 0.95)
CALIBRATION_REQUIRED_LEVELS = (0.90, 0.95)
QUANTILE_LEVELS = tuple(sorted({
    round(q, 12)
    for level in CALIBRATION_LEVELS
    for q in ((1 - level) / 2, (1 + level) / 2, level)
}))
PIT_BINS = 10
# A level passes when its coverage sits within this many standard errors of the
# level itself. In coverage mode, only the central 90% and 95% levels determine
# acceptance. In AD mode, all coverage levels remain descriptive diagnostics.
CALIBRATION_TOLERANCE_SE = 3.0

# Checkpoints go to storage3 alongside every other run; /home has a small quota.
DEFAULT_CHECKPOINT_DIR = DEFAULT_CKPT_ROOT / "wildfire_sensitivity"


ALL_COVARIATE_NAMES = (
    "P",
    "LAND",
    "PBLH^(1/5)",
    "CFLOW",
    "CFMID",
    "SWGDN",
    "SLP",
    "LWP^(1/5)",
    "WS",
    "LTS",
    "TS",
    "DUST^(1/5)",
    "OMEGA",
    "LWC^(1/5)",
    "RH",
    "T",
    "EPV",
    "CO^(1/5)",
    "PREC^(1/5)",
    "CHL^(1/2)",
    "DMS^(1/2)",
    "SO2EM^(1/5)",
)

# Expert-provided starting inputs; keep this order for reproducible fitting.
INITIAL_INSENSITIVE_COVARIATES = (
    "P", "LAND", "SLP", "WS", "TS", "DUST^(1/5)", "EPV",
    "CHL^(1/2)", "DMS^(1/2)", "SO2EM^(1/5)",
)
