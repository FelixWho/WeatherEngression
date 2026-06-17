"""Shared constants for synthetic engression experiments."""

QUANTILE_KEYS = ("q05", "q50", "q95")
QUANTILE_LEVELS = (0.05, 0.50, 0.95)

SPLIT_MODES = (
    "in-support",
    "right-extrapolation",
    "left-extrapolation",
    "two-sided-extrapolation",
)

DISPLAY_MODEL_NAMES = {
    "narx_gaussian": "NARX Gaussian",
    "narx_student_t": "NARX Student-t",
    "narx_garch": "NARX GARCH",
    "hurdle_lognormal": "Hurdle Lognormal",
    "regime_mixture": "Regime Mixture",
    "preadditive": "Preadditive",
}


def format_model_name(model: str) -> str:
    """Return a compact display label for a synthetic model name."""

    return DISPLAY_MODEL_NAMES.get(model, model.replace("_", " ").title())
