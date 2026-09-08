#!/usr/bin/env python3
"""
R2 reviewer-targeted experiments for the EJOR supply-chain manuscript.

The goal is not to reproduce every exploratory figure from the first revision.
This runner produces the extra evidence requested in R2:
  1. OR/RL benchmark comparisons, including pure RL without KG constraints.
  2. Digital-twin misspecification robustness.
  3. Action-space and runtime complexity summaries.
  4. Clear DataCo observed-variable vs inferred-proxy definitions.

Outputs are written below the directory supplied with ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"

DATA_PATH = Path(
    "/Users/malka/PycharmProjects/p_2025_SC_AI_EJOR/"
    "Dataset_SupplyChain/DataCoSupplyChainDataset.csv"
)

SEED = 20260602
ACTIONS = ["none", "expedite", "reroute", "contract", "review"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}


@dataclass
class EvalConfig:
    name: str
    delay_scale: float = 1.0
    delay_shift: float = 0.0
    trust_noise: float = 0.0
    heavy_tail_prob: float = 0.0
    gamma_kernel: bool = False
    missing_proxy_prob: float = 0.0


def ensure_dirs() -> None:
    for d in (RESULTS, TABLES, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def minmax(s: pd.Series) -> pd.Series:
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def load_and_build_state(sample_n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    if sample_n is not None and sample_n < len(df):
        df = df.sample(sample_n, random_state=SEED).reset_index(drop=True)

    required = [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Late_delivery_risk",
        "Delivery Status",
        "Order Region",
        "Market",
        "Customer Segment",
        "Shipping Mode",
        "Order Status",
        "Product Card Id",
        "Order Item Quantity",
        "Sales",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required DataCo columns: {missing}")

    out = df[required].copy()
    out["delay"] = (
        out["Days for shipping (real)"] - out["Days for shipment (scheduled)"]
    ).astype(float)
    out["positive_delay"] = out["delay"].clip(lower=0)
    out["late"] = out["Late_delivery_risk"].astype(float).clip(0, 1)

    status = out["Order Status"].astype(str).str.upper()
    delivery = out["Delivery Status"].astype(str).str.upper()
    out["adverse_status"] = (
        status.str.contains("FRAUD|CANCELED|CANCELLED|ON_HOLD|PAYMENT_REVIEW")
        | delivery.str.contains("CANCEL")
    ).astype(float)

    # Node proxy: DataCo does not have suppliers/manufacturers directly.
    # Product Card Id is used as an operational supply node proxy.
    out["node_id"] = out["Product Card Id"].astype(str)
    out["region"] = out["Order Region"].astype(str)
    out["segment"] = out["Customer Segment"].astype(str)
    out["market"] = out["Market"].astype(str)
    out["ship_mode"] = out["Shipping Mode"].astype(str)

    node = out.groupby("node_id").agg(
        node_delay_mean=("positive_delay", "mean"),
        node_late_rate=("late", "mean"),
        node_adverse_rate=("adverse_status", "mean"),
        node_volume=("node_id", "size"),
        node_sales=("Sales", "mean"),
    )
    out = out.join(node, on="node_id")

    delay_norm = minmax(out["node_delay_mean"])
    adverse_norm = minmax(out["node_adverse_rate"])
    late_norm = out["node_late_rate"].clip(0, 1)

    # Trust is an inferred reliability proxy, not a DataCo ground-truth field.
    reliability_raw = 1.0 - (0.55 * late_norm + 0.25 * delay_norm + 0.20 * adverse_norm)
    # Segment-wise z/logistic normalization for comparability.
    z = reliability_raw.groupby(out["market"]).transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    )
    out["trust_proxy"] = sigmoid(z.to_numpy())

    # Bias is an operational disparity proxy across observable groups.
    global_late = out["late"].mean()
    region_late = out.groupby("region")["late"].transform("mean")
    segment_late = out.groupby("segment")["late"].transform("mean")
    mode_late = out.groupby("ship_mode")["late"].transform("mean")
    disparity = pd.concat(
        [
            (region_late - global_late).abs(),
            (segment_late - global_late).abs(),
            (mode_late - global_late).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["bias_proxy"] = minmax(disparity).clip(0, 1)

    # Base risk score used by the digital-twin environment.
    out["delay_norm"] = minmax(out["positive_delay"]).clip(0, 1)
    out["volume_norm"] = minmax(np.log1p(out["node_volume"])).clip(0, 1)
    risk_logit = (
        -1.4
        + 2.1 * out["delay_norm"].to_numpy()
        + 1.6 * (1.0 - out["trust_proxy"].to_numpy())
        + 1.1 * out["bias_proxy"].to_numpy()
        + 1.0 * out["late"].to_numpy()
        + 0.45 * out["adverse_status"].to_numpy()
    )
    out["base_risk"] = sigmoid(risk_logit)

    # KG-feasibility attributes from observed operational conditions.
    region_counts = out.groupby("node_id")["region"].transform("nunique")
    out["has_alt_region"] = (region_counts > 1).astype(float)
    out["expedite_available"] = (out["ship_mode"].str.upper() != "SAME DAY").astype(float)
    out["contract_available"] = (out["node_volume"] >= out["node_volume"].quantile(0.25)).astype(float)
    out["regulatory_forbid"] = (
        out["adverse_status"].eq(1) & out["trust_proxy"].lt(0.35)
    ).astype(float)

    cols = [
        "node_id",
        "region",
        "market",
        "segment",
        "ship_mode",
        "delay",
        "delay_norm",
        "late",
        "adverse_status",
        "trust_proxy",
        "bias_proxy",
        "base_risk",
        "volume_norm",
        "has_alt_region",
        "expedite_available",
        "contract_available",
        "regulatory_forbid",
    ]
    return out[cols].dropna().reset_index(drop=True)


def state_matrix(df: pd.DataFrame) -> np.ndarray:
    x = df[
        [
            "delay_norm",
            "late",
            "trust_proxy",
            "bias_proxy",
            "base_risk",
            "volume_norm",
            "has_alt_region",
            "expedite_available",
            "contract_available",
            "regulatory_forbid",
        ]
    ].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(x, 0.0, 1.0), dtype=np.float64)


def kg_mask(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    mask = np.ones((n, len(ACTIONS)), dtype=bool)
    # none always technically possible
    mask[:, ACTION_TO_ID["expedite"]] = df["expedite_available"].to_numpy(dtype=bool)
    mask[:, ACTION_TO_ID["reroute"]] = (
        df["has_alt_region"].eq(1) & df["regulatory_forbid"].eq(0)
    ).to_numpy(dtype=bool)
    mask[:, ACTION_TO_ID["contract"]] = (
        df["contract_available"].eq(1) & df["regulatory_forbid"].eq(0)
    ).to_numpy(dtype=bool)
    mask[:, ACTION_TO_ID["review"]] = True
    return mask


def action_cost(actions: np.ndarray) -> np.ndarray:
    costs = np.array([0.00, 0.12, 0.20, 0.25, 0.10])
    return costs[actions]


def adjusted_outcomes(
    df: pd.DataFrame,
    actions: np.ndarray,
    cfg: EvalConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    base = df["base_risk"].to_numpy(float).copy()
    trust = df["trust_proxy"].to_numpy(float).copy()
    bias = df["bias_proxy"].to_numpy(float).copy()
    delay_norm = df["delay_norm"].to_numpy(float).copy()

    if cfg.gamma_kernel:
        shape = 2.0
        scale = np.maximum(delay_norm * cfg.delay_scale + 0.02, 0.02) / shape
        delay_norm = rng.gamma(shape, scale)
        delay_norm = np.clip(delay_norm, 0, 1)
    else:
        delay_norm = np.clip(delay_norm * cfg.delay_scale + cfg.delay_shift, 0, 1)

    if cfg.heavy_tail_prob > 0:
        shock = rng.random(len(df)) < cfg.heavy_tail_prob
        delay_norm[shock] = np.clip(delay_norm[shock] + rng.uniform(0.25, 0.65, shock.sum()), 0, 1)

    if cfg.trust_noise > 0:
        trust = np.clip(trust + rng.normal(0, cfg.trust_noise, len(df)), 0, 1)

    if cfg.missing_proxy_prob > 0:
        missing = rng.random(len(df)) < cfg.missing_proxy_prob
        trust[missing] = np.nanmean(trust)
        bias[missing] = np.nanmean(bias)

    risk = sigmoid(
        -1.4
        + 2.1 * delay_norm
        + 1.6 * (1.0 - trust)
        + 1.1 * bias
        + 1.0 * df["late"].to_numpy(float)
        + 0.45 * df["adverse_status"].to_numpy(float)
    )

    # Action effects: deliberately simple and transparent.
    effects = np.zeros(len(df))
    effects[actions == ACTION_TO_ID["expedite"]] = 0.22
    effects[actions == ACTION_TO_ID["reroute"]] = 0.30
    effects[actions == ACTION_TO_ID["contract"]] = 0.18
    effects[actions == ACTION_TO_ID["review"]] = 0.12

    # Review reduces bias most; contract improves trust; reroute/expedite reduce delay risk.
    bias_after = bias.copy()
    bias_after[actions == ACTION_TO_ID["review"]] *= 0.70
    bias_after[actions == ACTION_TO_ID["contract"]] *= 0.88

    risk_after = np.clip(risk * (1.0 - effects), 0.01, 0.99)

    # Trust loss is residual degradation pressure after mitigation, not a DataCo label.
    base_trust_loss = np.clip(0.15 * risk + 0.12 * bias + 0.08 * (trust < 0.45), 0, 1)
    trust_mitigation = np.zeros(len(df))
    trust_mitigation[actions == ACTION_TO_ID["expedite"]] = 0.10
    trust_mitigation[actions == ACTION_TO_ID["reroute"]] = 0.18
    trust_mitigation[actions == ACTION_TO_ID["contract"]] = 0.42
    trust_mitigation[actions == ACTION_TO_ID["review"]] = 0.32
    trust_loss = np.clip(base_trust_loss * (1.0 - trust_mitigation), 0, 1)
    trust_after = np.clip(trust - trust_loss, 0, 1)

    failures = rng.random(len(df)) < risk_after
    critical = failures & (delay_norm > 0.60) & (trust_after < 0.55)

    mask = kg_mask(df)
    infeasible = ~mask[np.arange(len(df)), actions]
    constraint_violation = infeasible

    return pd.DataFrame(
        {
            "region": df["region"].to_numpy(),
            "action": actions,
            "risk": risk_after,
            "critical": critical.astype(float),
            "cost": action_cost(actions),
            "trust_loss": trust_loss,
            "bias_after": bias_after,
            "constraint_violation": constraint_violation.astype(float),
        }
    )


def summarize_eval(out: pd.DataFrame, runtime_s: float, action_cardinality: float) -> dict:
    region_failure = out.groupby("region")["critical"].mean()
    disparity = float(region_failure.max() - region_failure.min()) if len(region_failure) else 0.0
    action_entropy = np.bincount(out["action"].to_numpy(), minlength=len(ACTIONS)) / len(out)
    action_entropy = float(-(action_entropy[action_entropy > 0] * np.log(action_entropy[action_entropy > 0])).sum())
    return {
        "CFR": out["critical"].mean(),
        "Cost": out["cost"].mean(),
        "Disparity": disparity,
        "TrustLoss": out["trust_loss"].mean(),
        "BiasProxyAfter": out["bias_after"].mean(),
        "ConstraintViolations": out["constraint_violation"].mean(),
        "ActionEntropy": action_entropy,
        "MeanFeasibleActions": action_cardinality,
        "RuntimeSeconds": runtime_s,
    }


def heuristic_actions(df: pd.DataFrame, name: str, rng: np.random.Generator) -> np.ndarray:
    risk = df["base_risk"].to_numpy(float)
    trust = df["trust_proxy"].to_numpy(float)
    bias = df["bias_proxy"].to_numpy(float)
    mask = kg_mask(df)
    n = len(df)
    actions = np.zeros(n, dtype=int)

    if name == "Static heuristic":
        actions[(df["late"].to_numpy(float) > 0.5) | (risk > 0.72)] = ACTION_TO_ID["expedite"]
    elif name == "Bayesian risk-only":
        actions[risk > 0.62] = ACTION_TO_ID["expedite"]
        actions[risk > 0.78] = ACTION_TO_ID["reroute"]
    elif name == "Digital-twin-only":
        actions[risk > 0.58] = ACTION_TO_ID["expedite"]
        actions[(risk > 0.72) & (trust < 0.55)] = ACTION_TO_ID["reroute"]
        actions[(risk > 0.82) & (trust < 0.45)] = ACTION_TO_ID["contract"]
    elif name == "Robust risk-averse":
        robust_risk = np.clip(risk + 0.20 * df["delay_norm"].to_numpy(float), 0, 1)
        actions[robust_risk > 0.50] = ACTION_TO_ID["expedite"]
        actions[robust_risk > 0.68] = ACTION_TO_ID["reroute"]
        actions[(robust_risk > 0.78) & (trust < 0.50)] = ACTION_TO_ID["review"]
    elif name == "Full KG-GPP":
        actions[risk > 0.55] = ACTION_TO_ID["expedite"]
        actions[(risk > 0.67) & (trust < 0.58)] = ACTION_TO_ID["reroute"]
        actions[(risk > 0.74) & (trust < 0.50)] = ACTION_TO_ID["contract"]
        actions[(bias > 0.55) | ((trust < 0.40) & (df["volume_norm"].to_numpy(float) < 0.25))] = ACTION_TO_ID["review"]
    else:
        raise ValueError(name)

    # Apply KG projection for KG-aware policies only.
    if name in {"Full KG-GPP", "Robust risk-averse", "Digital-twin-only"}:
        for i in range(n):
            if not mask[i, actions[i]]:
                feasible = np.flatnonzero(mask[i])
                # Prefer review for infeasible high-bias/low-trust cases.
                if ACTION_TO_ID["review"] in feasible:
                    actions[i] = ACTION_TO_ID["review"]
                else:
                    actions[i] = feasible[0]

    return actions


def gpp_expected_actions(df: pd.DataFrame, use_kg: bool = True) -> np.ndarray:
    """One-step model-based GPP action selection under the calibrated twin.

    This is deliberately transparent: each candidate action is scored by the
    same failure, cost, trust-loss, disparity, and violation terms reported in
    the benchmark table. The pure RL rows keep the identical state/action space
    but do not receive this model-based KG projection.
    """
    n = len(df)
    risk = df["base_risk"].to_numpy(float)
    trust = df["trust_proxy"].to_numpy(float)
    bias = df["bias_proxy"].to_numpy(float)
    delay = df["delay_norm"].to_numpy(float)
    mask = kg_mask(df)

    scores = np.zeros((n, len(ACTIONS)), dtype=float)
    for action_id in range(len(ACTIONS)):
        actions = np.full(n, action_id, dtype=int)
        effects = np.zeros(n)
        if action_id == ACTION_TO_ID["expedite"]:
            effects[:] = 0.22
        elif action_id == ACTION_TO_ID["reroute"]:
            effects[:] = 0.30
        elif action_id == ACTION_TO_ID["contract"]:
            effects[:] = 0.18
        elif action_id == ACTION_TO_ID["review"]:
            effects[:] = 0.12

        risk_after = np.clip(risk * (1.0 - effects), 0.01, 0.99)
        bias_after = bias.copy()
        if action_id == ACTION_TO_ID["review"]:
            bias_after *= 0.70
        elif action_id == ACTION_TO_ID["contract"]:
            bias_after *= 0.88

        base_trust_loss = np.clip(0.15 * risk + 0.12 * bias + 0.08 * (trust < 0.45), 0, 1)
        mitigation = np.zeros(n)
        if action_id == ACTION_TO_ID["expedite"]:
            mitigation[:] = 0.10
        elif action_id == ACTION_TO_ID["reroute"]:
            mitigation[:] = 0.18
        elif action_id == ACTION_TO_ID["contract"]:
            mitigation[:] = 0.42
        elif action_id == ACTION_TO_ID["review"]:
            mitigation[:] = 0.32
        trust_loss = np.clip(base_trust_loss * (1.0 - mitigation), 0, 1)
        trust_after = np.clip(trust - trust_loss, 0, 1)

        critical_proxy = risk_after * (delay > 0.60) * (trust_after < 0.55)
        infeasible = ~mask[:, action_id]
        high_risk_no_action = (risk > 0.72) & (actions == ACTION_TO_ID["none"])
        violation = infeasible | high_risk_no_action

        scores[:, action_id] = (
            1.45 * critical_proxy
            + 0.55 * action_cost(actions)
            + 0.38 * bias_after
            + 0.42 * trust_loss
            + 0.85 * violation.astype(float)
        )

    if use_kg:
        scores = np.where(mask, scores, 1e9)
    return scores.argmin(axis=1)


def train_linear_policy(
    train: pd.DataFrame,
    use_kg: bool,
    fairness_weight: float,
    epochs: int,
    batch_size: int,
    lr: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = state_matrix(train)
    n, d = x.shape
    x = np.ascontiguousarray(np.c_[np.ones(n, dtype=np.float64), x], dtype=np.float64)
    d += 1
    theta = rng.normal(0, 0.03, size=(d, len(ACTIONS)))
    mask = kg_mask(train)

    for _ in range(epochs):
        idx = rng.integers(0, n, size=batch_size)
        xb = x[idx]
        logits = np.einsum("bd,dk->bk", xb, theta)
        if use_kg:
            mb = mask[idx]
            logits = np.where(mb, logits, -1e9)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        actions = np.array([rng.choice(len(ACTIONS), p=p) for p in probs])
        batch = train.iloc[idx].reset_index(drop=True)
        out = adjusted_outcomes(batch, actions, EvalConfig("train"), rng)
        critical_weight = 1.0 + 0.75 * float(fairness_weight > 0)
        reward = (
            -critical_weight * out["critical"].to_numpy()
            -0.55 * out["cost"].to_numpy()
            -fairness_weight * out["bias_after"].to_numpy()
            -0.35 * out["trust_loss"].to_numpy()
            -0.75 * out["constraint_violation"].to_numpy()
        )
        # Normalize for REINFORCE stability.
        reward_std = reward.std()
        if reward_std < 1e-4:
            adv = reward - reward.mean()
        else:
            adv = (reward - reward.mean()) / reward_std
        adv = np.clip(np.nan_to_num(adv, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
        onehot = np.zeros_like(probs)
        onehot[np.arange(batch_size), actions] = 1.0
        grad_logits = (onehot - probs) * adv[:, None]
        grad = np.einsum("bd,bk->dk", xb, grad_logits) / batch_size
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 2.0:
            grad *= 2.0 / grad_norm
        theta += lr * grad
        theta = np.clip(np.nan_to_num(theta, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)

    return theta


def policy_actions(
    df: pd.DataFrame,
    theta: np.ndarray,
    use_kg: bool,
) -> np.ndarray:
    x = state_matrix(df)
    x = np.ascontiguousarray(np.c_[np.ones(len(df), dtype=np.float64), x], dtype=np.float64)
    logits = np.einsum("bd,dk->bk", x, theta)
    if use_kg:
        logits = np.where(kg_mask(df), logits, -1e9)
    return logits.argmax(axis=1)


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
        if name == "Full KG-GPP":
            actions = gpp_expected_actions(df, use_kg=True)
        else:
            actions = heuristic_actions(df, name, rng)
    else:
        actions = policy_actions(df, theta, use_kg=use_kg)
    out = adjusted_outcomes(df, actions, cfg, rng)
    runtime_s = time.perf_counter() - start
    action_card = kg_mask(df).sum(axis=1).mean() if use_kg or name in {"Full KG-GPP", "Robust risk-averse", "Digital-twin-only"} else len(ACTIONS)
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


def dataframe_to_latex(df: pd.DataFrame, path: Path, float_cols: int = 4) -> None:
    text = df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.{float_cols}f}")
    path.write_text(text)


def run(args: argparse.Namespace) -> None:
    global DATA_PATH, RESULTS, TABLES, LOGS
    DATA_PATH = Path(args.data_path).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    RESULTS = output_root / "results"
    TABLES = output_root / "tables"
    LOGS = output_root / "logs"
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    df = load_and_build_state(sample_n=args.sample_n)
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
        ("Digital-twin-only", None, True),
        ("Pure RL (no KG)", theta_pure, False),
        ("RL + KG (no fairness)", theta_kg, True),
        ("RL + fairness (no KG)", theta_fair, False),
        ("RL + KG + fairness", theta_full, True),
        ("Robust risk-averse", None, True),
        ("Full KG-GPP", None, True),
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

    # Robustness / misspecification requested by Reviewer 3.
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
        ("Full KG-GPP", None, True),
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
                "Operational definition": "Not used as a ground-truth label; bounded in manuscript to endogenous trust/disparity proxies and exogenous stress scenarios",
            },
        ]
    )

    bench.to_csv(RESULTS / "benchmark_results.csv", index=False)
    robust.to_csv(RESULTS / "misspecification_results.csv", index=False)
    complexity.to_csv(RESULTS / "complexity_results.csv", index=False)
    proxy_defs.to_csv(RESULTS / "dataco_proxy_definitions.csv", index=False)
    pd.concat(instance_rows, ignore_index=True).to_csv(
        RESULTS / "instance_results.csv.gz", index=False, compression="gzip"
    )

    dataframe_to_latex(bench, TABLES / "benchmark_results.tex")
    dataframe_to_latex(robust, TABLES / "misspecification_results.tex")
    dataframe_to_latex(complexity, TABLES / "complexity_results.tex")
    dataframe_to_latex(proxy_defs, TABLES / "dataco_proxy_definitions.tex")

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
    (LOGS / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print("Wrote:")
    for path in [
        RESULTS / "benchmark_results.csv",
        RESULTS / "misspecification_results.csv",
        RESULTS / "complexity_results.csv",
        RESULTS / "dataco_proxy_definitions.csv",
        RESULTS / "instance_results.csv.gz",
        TABLES / "benchmark_results.tex",
        TABLES / "misspecification_results.tex",
        TABLES / "complexity_results.tex",
        TABLES / "dataco_proxy_definitions.tex",
        LOGS / "run_metadata.json",
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
    ap.add_argument("--data-path", type=str, default=str(DATA_PATH), help="Path to DataCoSupplyChainDataset.csv")
    ap.add_argument("--output-dir", type=str, default=str(ROOT), help="Directory receiving results/, tables/, and logs/")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
