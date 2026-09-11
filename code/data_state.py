"""DataCo loading, operational proxy construction, and KG state utilities."""
import numpy as np
import pandas as pd
import config

def load_and_build_state(sample_n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PATH, encoding="ISO-8859-1")
    if sample_n is not None and sample_n < len(df):
        df = df.sample(sample_n, random_state=config.SEED).reset_index(drop=True)

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

    delay_norm = config.minmax(out["node_delay_mean"])
    adverse_norm = config.minmax(out["node_adverse_rate"])
    late_norm = out["node_late_rate"].clip(0, 1)

    # Trust is an inferred reliability proxy, not a DataCo ground-truth field.
    reliability_raw = 1.0 - (0.55 * late_norm + 0.25 * delay_norm + 0.20 * adverse_norm)
    # Segment-wise z/logistic normalization for comparability.
    z = reliability_raw.groupby(out["market"]).transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    )
    out["trust_proxy"] = config.sigmoid(z.to_numpy())

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
    out["bias_proxy"] = config.minmax(disparity).clip(0, 1)

    # Base risk score used by the digital-twin environment.
    out["delay_norm"] = config.minmax(out["positive_delay"]).clip(0, 1)
    out["volume_norm"] = config.minmax(np.log1p(out["node_volume"])).clip(0, 1)
    risk_logit = (
        -1.4
        + 2.1 * out["delay_norm"].to_numpy()
        + 1.6 * (1.0 - out["trust_proxy"].to_numpy())
        + 1.1 * out["bias_proxy"].to_numpy()
        + 1.0 * out["late"].to_numpy()
        + 0.45 * out["adverse_status"].to_numpy()
    )
    out["base_risk"] = config.sigmoid(risk_logit)

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
def generate_bayesian_risk_figure(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate and save the bounded trust/bias Bayesian risk diagnostic."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(config.SEED)
    n = 10_000
    trust_draws = config.sigmoid(rng.normal(df["trust_proxy"].mean(), df["trust_proxy"].std(), n))
    delay_draws = rng.normal(df["delay"].mean(), df["delay"].std(), n)
    bias_draws = config.sigmoid(rng.normal(df["bias_proxy"].mean(), df["bias_proxy"].std(), n))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, draws, label in zip(
        axes,
        (trust_draws, delay_draws, bias_draws),
        ("bounded trust proxy", "delay", "bounded bias proxy"),
    ):
        density, edges = np.histogram(draws, bins=120, density=True)
        ax.plot(0.5 * (edges[:-1] + edges[1:]), density, color="#4c72b0", linewidth=2)
        ax.set_title(label)
        ax.set_ylabel("")
        ax.text(0.5, 0.75, f"mean={draws.mean():.2f}", transform=ax.transAxes,
                ha="center", fontsize=14)
        if label != "delay":
            ax.set_xlim(0, 1)
            ax.set_xlabel("value in [0, 1]")
        else:
            ax.set_xlabel("days")
    fig.suptitle("Bayesian Inference of Supply Chain Risk Factors")
    fig.tight_layout()
    path = output_dir / "figures" / "Fig_Bayesian_Risk_Inference.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
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
    mask = np.ones((n, len(config.ACTIONS)), dtype=bool)
    # none always technically possible
    mask[:, config.ACTION_TO_ID["expedite"]] = df["expedite_available"].to_numpy(dtype=bool)
    mask[:, config.ACTION_TO_ID["reroute"]] = (
        df["has_alt_region"].eq(1) & df["regulatory_forbid"].eq(0)
    ).to_numpy(dtype=bool)
    mask[:, config.ACTION_TO_ID["contract"]] = (
        df["contract_available"].eq(1) & df["regulatory_forbid"].eq(0)
    ).to_numpy(dtype=bool)
    mask[:, config.ACTION_TO_ID["review"]] = True
    return mask
