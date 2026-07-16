"""T-learner: fit the wildfire / wildfire-free arms (train) and analyze them (test).

No package-level entry point or re-exports on purpose -- import from the specific
module you want:

    from experiments.t_learner.train import load_and_fit          # train + save
    from experiments.t_learner.test import load_saved, counterfactual_pair  # load + analyze

Run a module directly:

    python -m experiments.t_learner.train --recurrent-state-noise --device cuda ...
    python -m experiments.t_learner.test --checkpoint-dir /storage3/.../BB_criterion1_recurrent
"""
