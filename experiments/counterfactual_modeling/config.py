from dataclasses import dataclass, field

from engression_modifications.lstm import LSTMEngressionConfig


@dataclass
class CounterfactualConfig:
    experiment_random_seed: int = 2026
    lstm_architecture_and_training_config: LSTMEngressionConfig = field(
        default_factory=LSTMEngressionConfig
    )
    number_of_prediction_samples_per_observation: int = 400
    sensitive_covariates: tuple[str, ...] = (
        "CO^(1/5)",
        "LTS",
        "LWP^(1/5)",
        "SWGDN",
    )
