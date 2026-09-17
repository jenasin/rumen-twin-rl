"""State estimation from incomplete / noisy / delayed sensor observations.

Deliberately simple (three interchangeable methods, selected in config):

``hold``    last valid value is carried forward
``ema``     exponential smoothing, frozen while the sensor is unavailable
``kalman``  scalar Kalman filter with a random-walk process model

The Kalman variant is the only one that reports an *uncertainty*, which grows
while observations are missing:

    predict:   x_t^-  = x_{t-1}^+ ,           P_t^- = P_{t-1}^+ + Q
    update:    K_t    = P_t^- / (P_t^- + R)
               x_t^+  = x_t^- + K_t (z_t - x_t^-)
               P_t^+  = (1 - K_t) P_t^-
    missing:   x_t^+ = x_t^- ,  P_t^+ = P_t^-      (no correction)

Note a deliberate limitation: R is fixed at the *nominal* sensor variance.
When a scenario degrades the sensor (Scenario G) the filter is mis-specified,
exactly as a deployed estimator with a drifting sensor would be. Controllers
are not told that the sensor quality changed.

SYNTHETIC SIMULATION STUDY.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

from .sensors import Observation
from .utils import load_config


class ScalarEstimator:
    """One-dimensional estimator with support for missing observations."""

    def __init__(self, method: str, init_value: float, ema_alpha: float = 0.45,
                 process_var: float = 0.002, meas_var: float = 0.002,
                 init_var: float = 0.04):
        self.method = method
        self.ema_alpha = float(ema_alpha)
        self.Q = float(process_var)
        self.R = float(meas_var)
        self.init_var = float(init_var)
        self.reset(init_value)

    def reset(self, init_value: float) -> None:
        self.x = float(init_value)
        self.P = self.init_var
        self.steps_since_obs = 0
        self.n_obs = 0
        self.last_innovation = 0.0

    def update(self, z: Optional[float]) -> float:
        if self.method == "none":
            # no estimation: pass the raw reading through, hold on dropout
            if z is not None:
                self.x = float(z)
                self.steps_since_obs = 0
                self.n_obs += 1
            else:
                self.steps_since_obs += 1
            return self.x

        if self.method == "kalman":
            self.P = self.P + self.Q                      # predict
            if z is not None:
                K = self.P / (self.P + self.R)
                self.last_innovation = float(z) - self.x
                self.x = self.x + K * self.last_innovation
                self.P = (1.0 - K) * self.P
                self.steps_since_obs = 0
                self.n_obs += 1
            else:
                self.steps_since_obs += 1
            return self.x

        if self.method == "ema":
            if z is not None:
                self.x = (1.0 - self.ema_alpha) * self.x + self.ema_alpha * float(z)
                self.steps_since_obs = 0
                self.n_obs += 1
            else:
                self.steps_since_obs += 1
            return self.x

        # "hold"
        if z is not None:
            self.x = float(z)
            self.steps_since_obs = 0
            self.n_obs += 1
        else:
            self.steps_since_obs += 1
        return self.x

    @property
    def uncertainty(self) -> float:
        return float(np.sqrt(max(0.0, self.P))) if self.method == "kalman" else \
            float(0.05 * (1 + self.steps_since_obs))


@dataclass
class EstimatorOutput:
    """Everything the controller layer is allowed to know at step t."""

    values: Dict[str, float]
    pH_estimate: float
    pH_uncertainty: float
    pH_trend: float
    pH_min_recent: float
    frac_low_recent: float
    steps_since_pH_obs: int
    pH_available: bool


class RumenStateEstimator:
    """Bank of scalar estimators plus the derived history features."""

    CHANNELS = ["pH", "temperature", "rumination", "dry_matter_intake",
                "hydration_index", "heat_load", "milk_yield_proxy"]

    def __init__(self, config: Optional[Dict] = None, method: Optional[str] = None):
        self.cfg = config or load_config()
        ecfg = self.cfg["estimator"]
        self.method = method or ecfg["method"]
        self.trend_window = int(ecfg["trend_window"])
        self.history_window = int(ecfg["history_window"])
        self.low_thr = float(self.cfg["thresholds"]["low_pH"])
        kal = ecfg["kalman"]
        defaults = {k: self.cfg["state_variables"][v]["default"] for k, v in
                    [("pH", "rumen_pH"), ("temperature", "rumen_temperature"),
                     ("rumination", "rumination_minutes"),
                     ("dry_matter_intake", "dry_matter_intake"),
                     ("hydration_index", "hydration_index"),
                     ("heat_load", "heat_load"),
                     ("milk_yield_proxy", "milk_yield_proxy")]}
        self._defaults = defaults
        self._kal = kal
        self._ema_alpha = float(ecfg["ema_alpha"])
        self.est: Dict[str, ScalarEstimator] = {}
        self.reset()

    def reset(self, initial: Optional[Dict[str, float]] = None) -> None:
        init = dict(self._defaults)
        if initial:
            init.update(initial)
        self.est = {
            k: ScalarEstimator(self.method, init[k], self._ema_alpha,
                               self._kal["process_var"], self._kal["meas_var"],
                               self._kal["init_var"])
            for k in self.CHANNELS
        }
        self.pH_history: Deque[float] = deque(maxlen=self.history_window)
        self.trend_buf: Deque[float] = deque(maxlen=self.trend_window + 1)

    # ------------------------------------------------------------------
    def update(self, obs: Observation) -> EstimatorOutput:
        values = {k: self.est[k].update(obs.get(k)) for k in self.CHANNELS}
        pH_hat = values["pH"]
        self.pH_history.append(pH_hat)
        self.trend_buf.append(pH_hat)

        trend = 0.0
        if len(self.trend_buf) >= 2:
            trend = (self.trend_buf[-1] - self.trend_buf[0]) / (len(self.trend_buf) - 1)

        hist = np.asarray(self.pH_history, dtype=np.float64)
        return EstimatorOutput(
            values=values,
            pH_estimate=float(pH_hat),
            pH_uncertainty=float(self.est["pH"].uncertainty),
            pH_trend=float(trend),
            pH_min_recent=float(hist.min()),
            frac_low_recent=float((hist < self.low_thr).mean()),
            steps_since_pH_obs=int(self.est["pH"].steps_since_obs),
            pH_available=bool(obs.is_available("pH")),
        )
