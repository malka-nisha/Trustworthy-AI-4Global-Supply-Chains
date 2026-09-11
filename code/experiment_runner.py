"""Experiment orchestration for the KG-GPP EJOR reproducibility package."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import config
from config import ACTIONS, PROJECT_ROOT, SEED, EvalConfig, ensure_dirs
from data_state import load_and_build_state, generate_bayesian_risk_figure, kg_mask, state_matrix
from policies import heuristic_actions, gpp_expected_actions, train_linear_policy, policy_actions
from simulation import adjusted_outcomes, summarize_eval

def evaluate_policy(
    name: str,
    df: pd.DataFrame,
    cfg: EvalConfig,
    rng: np.random.Generator,
    theta: np.ndarray | None = None,
    use_kg: bool = False,
    return_instances: bool = False,
) -> dict | tuple[dict, pd.DataFrame]:
    start = time.perf_counter()
    if theta is None:
        if name == "Proposed KG-GPP":
            actions = gpp_expected_actions(df, use_kg=True)
        else:
            actions = heuristic_actions(df, name, rng)
    else:
        actions = policy_actions(df, theta, use_kg=use_kg)
    out = adjusted_outcomes(df, actions, cfg, rng)
    runtime_s = time.perf_counter() - start
    action_card = kg_mask(df).sum(axis=1).mean() if use_kg else len(ACTIONS)
    row = summarize_eval(out, runtime_s, action_card)
    row["Policy"] = name
    row["Scenario"] = cfg.name
    row["N"] = len(df)
    if return_instances:
        detail = out.copy()
        detail.insert(0, "InstanceID", np.arange(len(detail), dtype=int))
        detail.insert(1, "Policy", name)
        detail.insert(2, "Scenario", cfg.name)
        return row, detail
    return row
def paired_split(df: pd.DataFrame, eval_n: int, train_n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = rng.permutation(len(df))
    train_idx = idx[:train_n]
    eval_idx = idx[train_n : train_n + eval_n]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[eval_idx].reset_index(drop=True)
def run(args: argparse.Namespace) -> None:
    config.DATA_PATH = Path(args.data_path).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    config.RESULTS = output_root / "results"
    config.LOGS = output_root / "logs"
    config.FIGURES = output_root / "figures"
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    df = load_and_build_state(sample_n=args.sample_n)
    figure_path = generate_bayesian_risk_figure(df, output_root)
    train, ev = paired_split(df, eval_n=args.eval_n, train_n=args.train_n, rng=rng)

    # Train requested RL-style benchmarks with identical state/action spaces.
    t0 = time.perf_counter()
    theta_pure = train_linear_policy(train, use_kg=False, fairness_weight=0.0, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, rng=rng)
    theta_kg = train_linear_policy(train, use_kg=True, fairness_weight=0.0, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, rng=rng)
    theta_fair = train_linear_policy(train, use_kg=False, fairness_weight=0.6, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, rng=rng)
    theta_full = train_linear_policy(train, use_kg=True, fairness_weight=0.6, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, rng=rng)
    train_seconds = time.perf_counter() - t0

    policies = [
        ("Static heuristic", None, False),
        ("Bayesian risk-only", None, False),
        ("Digital-twin-only", None, False),
        ("Pure RL (no KG)", theta_pure, False),
        ("RL + KG (no fairness)", theta_kg, True),
        ("RL + fairness (no KG)", theta_fair, False),
        ("RL + KG + fairness", theta_full, True),
        ("Robust risk-averse", None, True),
        ("Proposed KG-GPP", None, True),
    ]

    base_cfg = EvalConfig("calibrated")
    rows = []
    instance_rows = []
    for name, theta, use_kg in policies:
        row, detail = evaluate_policy(name, ev, base_cfg, rng, theta=theta, use_kg=use_kg, return_instances=True)
        rows.append(row)
        instance_rows.append(detail)
    bench = pd.DataFrame(rows)
    bench = bench[
        [
            "Policy",
            "CFR",
            "Cost",
            "Disparity",
            "TrustLoss",
            "BiasProxyAfter",
            "ConstraintViolations",
            "MeanFeasibleActions",
            "RuntimeSeconds",
        ]
    ]

    # Evaluate robustness under alternative delay and proxy assumptions.
    scenarios = [
        EvalConfig("calibrated"),
        EvalConfig("gamma_delay_kernel", gamma_kernel=True),
        EvalConfig("heavy_tail_shocks", heavy_tail_prob=0.08),
        EvalConfig("delay_plus_10pct", delay_scale=1.10),
        EvalConfig("delay_plus_20pct", delay_scale=1.20),
        EvalConfig("delay_minus_10pct", delay_scale=0.90),
        EvalConfig("trust_noise_sd_0.05", trust_noise=0.05),
        EvalConfig("missing_proxy_10pct", missing_proxy_prob=0.10),
    ]
    robust_rows = []
    robust_policies = [
        ("Pure RL (no KG)", theta_pure, False),
        ("RL + KG (no fairness)", theta_kg, True),
        ("RL + KG + fairness", theta_full, True),
        ("Proposed KG-GPP", None, True),
        ("Robust risk-averse", None, True),
    ]
    for cfg in scenarios:
        for name, theta, use_kg in robust_policies:
            row, detail = evaluate_policy(name, ev, cfg, rng, theta=theta, use_kg=use_kg, return_instances=True)
            robust_rows.append(row)
            instance_rows.append(detail)
    robust = pd.DataFrame(robust_rows)
    robust = robust[
        [
            "Scenario",
            "Policy",
            "CFR",
            "Cost",
            "Disparity",
            "ConstraintViolations",
            "BiasProxyAfter",
        ]
    ]

    # Complexity evidence across sample sizes.
    complexity_rows = []
    for n in [2_500, 5_000, 10_000, min(20_000, len(df)), min(40_000, len(df))]:
        sub = df.sample(n, random_state=SEED + n).reset_index(drop=True)
        start = time.perf_counter()
        m = kg_mask(sub)
        projection_s = time.perf_counter() - start
        nodes = sub["node_id"].nunique()
        regions = sub["region"].nunique()
        raw_actions = n * len(ACTIONS)
        feasible_actions = int(m.sum())
        xdim = state_matrix(sub).shape[1]
        memory_mb = (state_matrix(sub).nbytes + m.nbytes) / (1024 * 1024)
        complexity_rows.append(
            {
                "Rows": n,
                "Nodes": nodes,
                "Regions": regions,
                "StateDim": xdim,
                "RawActions": raw_actions,
                "KGFeasibleActions": feasible_actions,
                "ActionReductionPct": 100 * (1 - feasible_actions / raw_actions),
                "MeanFeasibleActions": feasible_actions / n,
                "ProjectionSeconds": projection_s,
                "StateMaskMemoryMB": memory_mb,
            }
        )
    complexity = pd.DataFrame(complexity_rows)

    proxy_defs = pd.DataFrame(
        [
            {
                "Construct": "Observed delivery delay",
                "Data status": "Directly observed in DataCo",
                "Operational definition": "Days for shipping (real) minus Days for shipment (scheduled)",
            },
            {
                "Construct": "Late-delivery flag",
                "Data status": "Directly observed in DataCo",
                "Operational definition": "Late_delivery_risk field",
            },
            {
                "Construct": "Trust",
                "Data status": "Inferred proxy, not ground truth",
                "Operational definition": "Logistic-normalized reliability score from node late rate, positive delay, and adverse/cancelled/fraud order status",
            },
            {
                "Construct": "Bias/disparity",
                "Data status": "Inferred operational proxy, not ground truth",
                "Operational definition": "Maximum absolute gap in late-delivery rate across observed region, customer-segment, and shipping-mode groups",
            },
            {
                "Construct": "Ideological risk",
                "Data status": "Not directly observed in DataCo",
                "Operational definition": "Not used as a ground-truth label; represented through endogenous trust/disparity proxies and exogenous stress scenarios",
            },
        ]
    )

    bench.to_csv(config.RESULTS / "benchmark_results.csv", index=False)
    robust.to_csv(config.RESULTS / "robustness_results.csv", index=False)
    complexity.to_csv(config.RESULTS / "complexity_results.csv", index=False)
    proxy_defs.to_csv(config.RESULTS / "dataco_proxy_definitions.csv", index=False)
    pd.concat(instance_rows, ignore_index=True).to_csv(
        config.RESULTS / "instance_results.csv.gz", index=False, compression="gzip"
    )

    meta = {
        "seed": SEED,
        "data_path": Path(args.data_path).name,
        "rows_loaded": len(df),
        "train_n": len(train),
        "eval_n": len(ev),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "rl_training_seconds_total": train_seconds,
        "policies": [p[0] for p in policies],
        "scenarios": [s.name for s in scenarios],
    }
    (config.LOGS / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print("Wrote:")
    for path in [
        config.RESULTS / "benchmark_results.csv",
        config.RESULTS / "robustness_results.csv",
        config.RESULTS / "complexity_results.csv",
        config.RESULTS / "dataco_proxy_definitions.csv",
        config.RESULTS / "instance_results.csv.gz",
        config.LOGS / "run_metadata.json",
        figure_path,
    ]:
        print(" -", path)
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--train-n", type=int, default=60_000)
    ap.add_argument("--eval-n", type=int, default=25_000)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=0.012)
    ap.add_argument("--data-path", type=str, default=str(config.DATA_PATH), help="Path to DataCoSupplyChainDataset.csv")
    ap.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT), help="Repository root receiving results/, logs/, and figures/")
    return ap.parse_args()
