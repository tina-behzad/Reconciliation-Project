# Reconciliation-Project
This is an empirical study on the paper: Reconciling Individual Probability Forecasts by Aaron Roth, Alexander Tolbert, and Scott Weinstein.
In this project, we will implement the algorithms discussed in the paper and analyze how they perform in a real scenario.

The main branch includes the pipeline for running Reconcile.Pipeline consists of:
1. Finding similar models with significant disagreements.
2. Running Reconcile

In the main branch, comparison.py includes experiments for comparing Reconcile with methods suggested in section 6 from "Model multiplicity: Opportunities, concerns, and solutions" (Black, Raghavan,
and Barocas 2022). This experiments focus on choosing one model from the set of models with similar accuracy.

The experiment branch includes experiments on quantifying the severity of predictive multiplicity in a set.