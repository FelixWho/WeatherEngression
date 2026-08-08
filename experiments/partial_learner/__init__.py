"""Partial-learner: a clean-data arm vs an ALL-DATA arm.

Sibling of ``experiments.t_learner``. The ONLY difference is the second model:
here it is trained on ALL data (clean + wildfire) instead of wildfire-only. The
clean arm is identical to the T-learner's clean arm, so the two approaches can be
compared directly.

Import from the specific module:

    from experiments.partial_learner.train import load_and_fit
    from experiments.partial_learner.test import load_saved

Run a module directly:

    python -m experiments.partial_learner.train --recurrent-state-noise --device cuda ...
    python -m experiments.partial_learner.test --checkpoint-dir /storage3/.../...

Note: contrasting ``all(x) - clean(x)`` gives a BIASED (attenuated-toward-zero)
effect estimate compared with the T-learner, because the all-data arm is a blend
of the wildfire and clean conditionals. This folder exists to quantify that gap.
"""
