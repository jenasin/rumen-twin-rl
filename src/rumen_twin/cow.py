"""Virtual dairy-cow population with hierarchical parameter variability.

Hierarchy
---------
    population hyperparameters  (config['population']['hyper'])
        -> individual cow parameters   (drawn once, fixed per cow)
            -> within-cow stochastic variation (per simulation step, in physiology.py)

Traits are drawn through a Gaussian copula so that a few biologically
sensible correlations (e.g. milk potential with intake capacity) are present.

SYNTHETIC SIMULATION STUDY — the parameter distributions are illustrative,
not estimates from real herds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import load_config, load_seeds, make_rng

TRAITS: List[str] = [
    "baseline_pH",
    "concentrate_sensitivity",
    "rumination_efficiency",
    "acid_clearance",
    "buffer_response",
    "intake_capacity_kg_day",
    "heat_sensitivity",
    "recovery_ability",
    "milk_potential_kg_day",
]


@dataclass
class Cow:
    """Individual virtual cow: an immutable bundle of physiological traits."""

    cow_id: int
    split: str
    baseline_pH: float
    concentrate_sensitivity: float
    rumination_efficiency: float
    acid_clearance: float
    buffer_response: float
    intake_capacity_kg_day: float
    heat_sensitivity: float
    recovery_ability: float
    milk_potential_kg_day: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, t) for t in TRAITS], dtype=np.float64)


def _correlation_matrix(cfg: Dict) -> np.ndarray:
    """Build a valid correlation matrix from the pairwise entries in config."""
    n = len(TRAITS)
    idx = {t: i for i, t in enumerate(TRAITS)}
    C = np.eye(n)
    for key, rho in (cfg["population"].get("trait_correlations") or {}).items():
        a, b = key.split("__")
        if a in idx and b in idx:
            C[idx[a], idx[b]] = C[idx[b], idx[a]] = float(rho)
    # Project to the nearest positive-definite matrix (eigenvalue clipping).
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    C = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    return C


class CowPopulation:
    """A reproducible population of synthetic cows with a train/val/test split.

    The split is applied to *individuals*, so test cows have their own trait
    draws that the RL agents never encounter during training.
    """

    def __init__(self, config: Optional[Dict] = None, seeds: Optional[Dict] = None,
                 n_cows: Optional[int] = None, homogeneous: bool = False):
        self.config = config or load_config()
        self.seeds = seeds or load_seeds()
        self.n_cows = int(n_cows or self.config["population"]["n_cows"])
        # ``homogeneous=True`` removes individual variability (Ablation 2):
        # every cow gets the population mean trait vector.
        self.homogeneous = bool(homogeneous)
        self.cows: List[Cow] = []
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        cfg = self.config
        hyper = cfg["population"]["hyper"]
        rng = make_rng(self.seeds["population"]["cow_generation"])

        C = _correlation_matrix(cfg)
        L = np.linalg.cholesky(C)
        z = rng.standard_normal((self.n_cows, len(TRAITS))) @ L.T  # correlated N(0,1)

        traits = {}
        for j, t in enumerate(TRAITS):
            mean, sd, lo, hi = hyper[t]
            if self.homogeneous:
                vals = np.full(self.n_cows, float(mean))
            else:
                vals = np.clip(mean + sd * z[:, j], lo, hi)
            traits[t] = vals

        splits = self._assign_splits()
        self.cows = [
            Cow(cow_id=i, split=splits[i], **{t: float(traits[t][i]) for t in TRAITS})
            for i in range(self.n_cows)
        ]

    def _assign_splits(self) -> np.ndarray:
        frac = self.config["population"]["split"]
        rng = make_rng(self.seeds["population"]["split"])
        perm = rng.permutation(self.n_cows)
        n_tr = int(round(frac["train"] * self.n_cows))
        n_va = int(round(frac["validation"] * self.n_cows))
        labels = np.empty(self.n_cows, dtype=object)
        labels[perm[:n_tr]] = "train"
        labels[perm[n_tr:n_tr + n_va]] = "validation"
        labels[perm[n_tr + n_va:]] = "test"
        return labels

    # ------------------------------------------------------------------
    def subset(self, split: str) -> List[Cow]:
        if split in (None, "all"):
            return list(self.cows)
        return [c for c in self.cows if c.split == split]

    def get(self, cow_id: int) -> Cow:
        return self.cows[int(cow_id)]

    def sample(self, split: str, rng: np.random.Generator) -> Cow:
        pool = self.subset(split)
        return pool[int(rng.integers(len(pool)))]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_dict() for c in self.cows])

    def summary(self) -> pd.DataFrame:
        df = self.to_frame()
        out = df.groupby("split")[TRAITS].agg(["mean", "std"]).round(3)
        return out

    def __len__(self) -> int:
        return len(self.cows)

    def __repr__(self) -> str:
        counts = self.to_frame()["split"].value_counts().to_dict()
        return f"CowPopulation(n={self.n_cows}, splits={counts}, homogeneous={self.homogeneous})"
