"""Synthetic sensor layer: the only channel through which a controller sees
the twin.

This module is what creates the *partial observability* the study is about:

  * three state variables (``vfa_index``, ``lactate_index``, ``health_risk``)
    have **no sensor at all** and are permanently latent;
  * the remaining sensors add Gaussian noise, an optional systematic bias
    (with optional random-walk drift), stochastic dropout and transport delay.

A delayed sensor is modelled physically: the noisy reading is *generated* at
the moment of measurement and *released* ``delay_steps`` later, so a delayed
observation is a genuinely stale measurement, not a smoothed one.

SYNTHETIC SIMULATION STUDY.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from .physiology import TwinState
from .utils import load_config

# sensor key -> state attribute
SENSOR_TO_STATE: Dict[str, str] = {
    "pH": "rumen_pH",
    "temperature": "rumen_temperature",
    "rumination": "rumination_minutes",
    "dry_matter_intake": "dry_matter_intake",
    "hydration_index": "hydration_index",
    "heat_load": "heat_load",
    "milk_yield_proxy": "milk_yield_proxy",
}

LATENT_VARIABLES: List[str] = ["vfa_index", "lactate_index", "health_risk"]


@dataclass
class Observation:
    """One multivariate sensor reading. ``None`` means 'not available'."""

    values: Dict[str, Optional[float]] = field(default_factory=dict)
    available: Dict[str, bool] = field(default_factory=dict)
    step_index: int = 0

    def get(self, key: str) -> Optional[float]:
        return self.values.get(key)

    def is_available(self, key: str) -> bool:
        return bool(self.available.get(key, False))

    def as_dict(self) -> Dict[str, Optional[float]]:
        return dict(self.values)


class _Channel:
    """A single synthetic sensor channel."""

    def __init__(self, key: str, spec: Dict, rng: np.random.Generator, n_steps: int):
        self.key = key
        self.rng = rng
        self.noise_sd = float(spec.get("noise_sd", 0.0))
        self.bias0 = float(spec.get("bias", 0.0))
        self.bias_drift_sd = float(spec.get("bias_drift_sd", 0.0))
        self.dropout_prob = float(spec.get("dropout_prob", 0.0))
        self.delay_steps = int(spec.get("delay_steps", 0))
        mask = spec.get("dropout_mask", None)
        self.dropout_mask = np.asarray(mask, dtype=bool) if mask is not None else None
        self.n_steps = n_steps
        self.reset()

    def reset(self) -> None:
        self.bias = self.bias0
        self.buffer: Deque[Optional[float]] = deque(maxlen=max(1, self.delay_steps + 1))
        # a delayed sensor has no data to release for its first `delay` steps
        for _ in range(self.delay_steps):
            self.buffer.append(None)

    def read(self, true_value: float, t: int, noise_scale: float = 1.0,
             dropout_scale: float = 1.0) -> Optional[float]:
        """Generate the reading taken at time ``t`` and release the due one."""
        if self.bias_drift_sd > 0.0:
            self.bias += float(self.rng.normal(0.0, self.bias_drift_sd))

        dropped = False
        if self.dropout_mask is not None and t < len(self.dropout_mask):
            dropped = bool(self.dropout_mask[t])
        if not dropped:
            p = min(1.0, self.dropout_prob * dropout_scale)
            dropped = bool(self.rng.random() < p)

        if dropped:
            reading: Optional[float] = None
        else:
            sd = self.noise_sd * noise_scale
            reading = float(true_value + self.bias +
                            (self.rng.normal(0.0, sd) if sd > 0 else 0.0))

        self.buffer.append(reading)
        if self.delay_steps <= 0:
            return reading
        # buffer holds delay+1 entries; the oldest is the one now due
        return self.buffer[0]


class SensorArray:
    """The full set of synthetic sensors attached to one cow for one episode."""

    def __init__(self, config: Optional[Dict] = None,
                 rng: Optional[np.random.Generator] = None,
                 overrides: Optional[Dict[str, Dict]] = None,
                 n_steps: int = 192):
        self.cfg = config or load_config()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.n_steps = int(n_steps)
        self.overrides = overrides or {}
        self.channels: Dict[str, _Channel] = {}
        self._build()

    def _build(self) -> None:
        for key, base_spec in self.cfg["sensors"].items():
            if not base_spec.get("observable", True):
                continue
            spec = dict(base_spec)
            spec.update(self.overrides.get(key, {}))
            self.channels[key] = _Channel(key, spec, self.rng, self.n_steps)

    def reset(self, rng: Optional[np.random.Generator] = None,
              overrides: Optional[Dict[str, Dict]] = None) -> None:
        if rng is not None:
            self.rng = rng
        if overrides is not None:
            self.overrides = overrides
        self.channels.clear()
        self._build()

    # ------------------------------------------------------------------
    def observe(self, state: TwinState, t: int, noise_scale: float = 1.0,
                dropout_scale: float = 1.0) -> Observation:
        """Produce the observation available to the controller at step ``t``."""
        values: Dict[str, Optional[float]] = {}
        available: Dict[str, bool] = {}
        for key, ch in self.channels.items():
            truth = float(getattr(state, SENSOR_TO_STATE[key]))
            v = ch.read(truth, t, noise_scale=noise_scale, dropout_scale=dropout_scale)
            values[key] = v
            available[key] = v is not None
        # latent variables are never observable
        for key in LATENT_VARIABLES:
            values[key] = None
            available[key] = False
        return Observation(values=values, available=available, step_index=int(t))

    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Dict[str, float]]:
        return {
            k: {"noise_sd": c.noise_sd, "bias": c.bias0,
                "dropout_prob": c.dropout_prob, "delay_steps": c.delay_steps}
            for k, c in self.channels.items()
        }
