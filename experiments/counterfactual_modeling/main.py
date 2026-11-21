from pathlib import Path
import sys


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.counterfactual_modeling.config import CounterfactualConfig


def train_sensitive_covariate_model(config: CounterfactualConfig):
    """Model 1: predict sensitive covariates from robust covariates and background emissions.

    Train only on clean observations. During wildfire periods, generate the
    sensitive covariates that could have occurred without wildfire.
    """
    pass


def train_clean_ccn_model(config: CounterfactualConfig):
    """Model 2: predict CCN at arrival without wildfire.

    Train only on clean observations, using robust/sensitive covariates and
    background emissions. Each draw predicts one CCN value at the observation
    site. At counterfactual prediction time, use Model 1's generated sensitive
    covariates.
    """
    pass


def train_wildfire_ccn_model(config: CounterfactualConfig):
    """Model 3: predict CCN at arrival with wildfire present.

    Use observed robust/sensitive covariates, background emissions, and wildfire
    emissions. Train primarily on wildfire observations, with clean observations
    included at lower weight for stability. Each draw predicts one CCN value at
    the observation site.
    """
    pass


def main(config: CounterfactualConfig):
    train_sensitive_covariate_model(config)
    train_clean_ccn_model(config)
    train_wildfire_ccn_model(config)


if __name__ == "__main__":
    main(CounterfactualConfig())
