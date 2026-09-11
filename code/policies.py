"""Heuristic, model-based KG-GPP, and policy-gradient comparators."""
import numpy as np
import pandas as pd
from config import ACTION_TO_ID, ACTIONS, EvalConfig
from data_state import kg_mask, state_matrix
from simulation import action_cost, adjusted_outcomes

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
    else:
        raise ValueError(name)

    # Apply KG projection for KG-aware policies only.
    if name in {"Proposed KG-GPP", "Robust risk-averse"}:
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

    Each candidate action is scored using expected critical outcomes,
    intervention cost, residual trust loss, the operational bias/disparity
    proxy, and feasibility-violation penalties. This action-selection score
    is distinct from the aggregate evaluation objective reported in the
    manuscript.
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
