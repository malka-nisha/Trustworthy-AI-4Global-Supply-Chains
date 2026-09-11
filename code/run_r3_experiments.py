#!/usr/bin/env python3
"""Run the KG-GPP EJOR reproducibility experiments.

The empirical package is split into focused modules: data/state construction,
digital-twin simulation, policy comparators, and experiment orchestration.
The manuscript presents the broader sequential KG-constrained POMDP; this code
implements the evaluated one-step KG-GPP benchmark.
"""
from experiment_runner import parse_args, run

if __name__ == "__main__":
    run(parse_args())
