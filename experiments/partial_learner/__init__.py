"""Partial-learner: a clean arm against an all-data arm.

Sibling of ``experiments.t_learner``, differing in one place: the second model is fit
on everything (clean + wildfire) rather than on wildfire alone. The clean arm is the
same either way, so the two approaches line up for comparison.

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
