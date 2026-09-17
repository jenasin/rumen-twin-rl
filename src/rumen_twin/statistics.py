"""Statistical summaries for repeated simulation outputs (Section 15).

Design choices worth stating explicitly, because they matter for how the
results should be read:

* **Everything is paired.** All controllers are evaluated on an identical
  ``(cow, scenario, episode seed)`` plan, so the natural unit of analysis is
  the *difference within a matched simulated episode*, not two independent
  samples.

* **p-values are reported but deliberately de-emphasised.** In a simulation
  study the sample size is a budget decision, not a property of nature: any
  non-zero difference can be made "significant" by simulating more episodes.
  We therefore lead with effect sizes and bootstrap confidence intervals and
  treat the paired Wilcoxon test only as a check that a difference is
  consistent in sign across episodes, never as the headline result.

* **Effect sizes** are the paired Hedges' g (bias-corrected standardised mean
  difference of the within-pair differences) and the matched-pairs
  rank-biserial correlation, which is the paired analogue of Cliff's delta and
  makes no normality assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
except ImportError:  # pragma: no cover
    sps = None


# ---------------------------------------------------------------------------
def describe(x: Sequence[float]) -> Dict[str, float]:
    """mean / SD / median / IQR summary of one metric."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: float("nan") for k in
                ["n", "mean", "sd", "median", "q1", "q3", "iqr", "min", "max"]}
    q1, q3 = np.percentile(a, [25, 75])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "min": float(a.min()), "max": float(a.max()),
    }


def bootstrap_ci(x: Sequence[float], statistic: Callable = np.mean,
                 n_resamples: int = 5000, level: float = 0.95,
                 seed: int = 12345) -> Dict[str, float]:
    """Percentile bootstrap CI for a statistic of one sample."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2:
        v = float(statistic(a)) if a.size else float("nan")
        return {"estimate": v, "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(int(n_resamples), a.size))
    boot = statistic(a[idx], axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return {"estimate": float(statistic(a)), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float],
                          n_resamples: int = 5000, level: float = 0.95,
                          seed: int = 12345) -> Dict[str, float]:
    """Bootstrap CI for the mean of the within-pair differences (a - b)."""
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    d = x - y
    d = d[np.isfinite(d)]
    out = bootstrap_ci(d, np.mean, n_resamples, level, seed)
    return {"mean_difference": out["estimate"],
            "ci_low": out["ci_low"], "ci_high": out["ci_high"], "n_pairs": int(d.size)}


# ---------------------------------------------------------------------------
def hedges_g_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Bias-corrected standardised mean difference of the paired differences."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0
    g = d.mean() / sd
    return float(g * (1.0 - 3.0 / (4.0 * n - 5.0)))      # small-sample correction


def matched_rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation in [-1, 1].

    +1 means every episode improved, -1 every episode worsened. Paired,
    distribution-free analogue of Cliff's delta.
    """
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(d)) if sps is not None else np.argsort(np.argsort(np.abs(d))) + 1.0
    total = ranks.sum()
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / total)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Unpaired Cliff's delta (reported alongside for reference)."""
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    # rank-based O(n log n) formulation
    allv = np.concatenate([x, y])
    r = sps.rankdata(allv) if sps is not None else np.argsort(np.argsort(allv)) + 1.0
    rx = r[:x.size].sum()
    u = rx - x.size * (x.size + 1) / 2.0
    return float(2.0 * u / (x.size * y.size) - 1.0)


def interpret_effect(g: float) -> str:
    a = abs(g)
    if not np.isfinite(a):
        return "undefined"
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


# ---------------------------------------------------------------------------
@dataclass
class PairedComparison:
    metric: str
    group_a: str
    group_b: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_difference: float
    ci_low: float
    ci_high: float
    hedges_g_paired: float
    effect_magnitude: str
    matched_rank_biserial: float
    cliffs_delta: float
    wilcoxon_statistic: float
    wilcoxon_p: float
    pct_pairs_favouring_a: float
    pct_pairs_tied: float
    pct_pairs_favouring_b: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def compare_paired(df: pd.DataFrame, metric: str, group_a: str, group_b: str,
                   group_col: str = "controller",
                   pair_keys: Sequence[str] = ("scenario", "eval_seed", "episode_index"),
                   n_resamples: int = 5000, level: float = 0.95,
                   seed: int = 12345, higher_is_better: bool = True) -> PairedComparison:
    """Paired comparison of two controllers on one metric.

    Rows are matched on ``pair_keys`` (which identify the identical simulated
    episode), so the comparison removes cow- and disturbance-level variance.
    """
    keys = list(pair_keys)
    # boolean metrics (e.g. episode success) must be numeric before differencing
    a = df[df[group_col] == group_a].set_index(keys)[metric].astype(float)
    b = df[df[group_col] == group_b].set_index(keys)[metric].astype(float)
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if joined.empty:
        raise ValueError(f"No matched episodes for {group_a} vs {group_b} on '{metric}'")
    x, y = joined["a"].to_numpy(), joined["b"].to_numpy()

    boot = paired_bootstrap_diff(x, y, n_resamples, level, seed)
    g = hedges_g_paired(x, y)
    if sps is not None and np.any(x - y != 0):
        try:
            w = sps.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            wstat, wp = float(w.statistic), float(w.pvalue)
        except ValueError:
            wstat, wp = float("nan"), float("nan")
    else:
        wstat, wp = float("nan"), float("nan")

    d = x - y
    # Report ties explicitly. On metrics with a floor (e.g. episodes where both
    # controllers reach 0 % low-pH time) ties are common, and quoting only the
    # "better in X %" figure would understate a one-sided result.
    better = (d > 0) if higher_is_better else (d < 0)
    worse = (d < 0) if higher_is_better else (d > 0)
    tied = (d == 0)
    return PairedComparison(
        metric=metric, group_a=group_a, group_b=group_b, n_pairs=int(len(x)),
        mean_a=float(x.mean()), mean_b=float(y.mean()),
        mean_difference=boot["mean_difference"], ci_low=boot["ci_low"], ci_high=boot["ci_high"],
        hedges_g_paired=g, effect_magnitude=interpret_effect(g),
        matched_rank_biserial=matched_rank_biserial(x, y),
        cliffs_delta=cliffs_delta(x, y), wilcoxon_statistic=wstat, wilcoxon_p=wp,
        pct_pairs_favouring_a=float(100.0 * better.mean()),
        pct_pairs_tied=float(100.0 * tied.mean()),
        pct_pairs_favouring_b=float(100.0 * worse.mean()),
    )


def summarise_metric(df: pd.DataFrame, metric: str, group_col: str = "controller",
                     n_resamples: int = 5000, level: float = 0.95,
                     seed: int = 12345) -> pd.DataFrame:
    """mean / SD / median / IQR / 95 % bootstrap CI of a metric, per group."""
    rows = []
    for g, sub in df.groupby(group_col, sort=False):
        d = describe(sub[metric])
        ci = bootstrap_ci(sub[metric], np.mean, n_resamples, level, seed)
        rows.append({group_col: g, "metric": metric, **d,
                     "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
    return pd.DataFrame(rows)
