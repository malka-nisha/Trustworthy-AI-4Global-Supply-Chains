"""Shared configuration and output paths for KG-GPP experiments."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
RESULTS = PROJECT_ROOT / "results"
LOGS = PROJECT_ROOT / "logs"
FIGURES = PROJECT_ROOT / "figures"
DATA_PATH = PROJECT_ROOT / "data" / "DataCoSupplyChainDataset.csv"
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
    for d in (RESULTS, LOGS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)

def minmax(s: pd.Series) -> pd.Series:
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
