# Testing for Sensitive Variables

## Motivation

We aim to understand which variables are mediators of wildfire effects. For example, wildfire may impact the temperature covariate, which in turn impacts CCN count.

Once we have the set of insensitive versus sensitive covariates, $M, M'$ respectively, we can predict the values for $M'$ in the absence of wildfires. This is the **counterfactual**.

## Stategy

We start with an insensitive set determined by weather experts (Shengqian, Wang), $M$. The remaining covariates, $M'$ are assumed to be wildfire-sensitive until we have evidence otherwise.

Imagine $M = \{A, B, C\}$, $M' = \{D, E, F\}$. One by one consider $D, E, F$ to see if they should be moved into $M$. How?



**Question:** does order of consideration of $D,E,F$ matter? Maxine believes no, but should double-check with permutation testing.