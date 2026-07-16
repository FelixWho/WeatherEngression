# archive/

Parked, superseded code kept for reference. Nothing here is imported by the
live pipeline (`experiments/`, `engression_modifications/`).

- `engression_model_starter/`: the original prototype (`energy.py`,
  `networks.py`) written before the real models existed. It sketched a
  pre-additive MLP generator and a hand-rolled energy loss. Both were replaced:
  the real models live in [`engression_modifications/`](../engression_modifications/)
  and the energy loss comes from the upstream `engression` package. Kept only so
  the early design notes aren't lost; safe to `git rm` if you don't want it.
