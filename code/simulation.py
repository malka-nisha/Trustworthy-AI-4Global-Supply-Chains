"""Digital-twin outcome simulation and evaluation summaries."""
import numpy as np
import pandas as pd
from config import ACTION_TO_ID, ACTIONS, EvalConfig, sigmoid
from data_state import kg_mask

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

    # Action-effect parameters used in the simulation model.
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
