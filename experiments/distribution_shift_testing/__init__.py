"""Checks on whether the test-period weather looks like what we trained on.

Two tests, both working off the per-channel summaries in ``features``:

* ``c2st`` asks whether train and test are even distinguishable, and if so which
  channels give it away. Reports an AUC rather than a p-value.
* ``knn_overlap`` asks the harder question: are test points somewhere the training
  data covers, or is the model extrapolating? This is what ATT leans on.

Run them together with ``run.py``; ``run_mmd.py`` adds the formal test.
"""
