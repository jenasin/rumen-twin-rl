"""Synthetic sensor behaviour: noise, bias, dropout, delay and latency."""
import numpy as np
import pytest

from rumen_twin.physiology import TwinState
from rumen_twin.sensors import LATENT_VARIABLES, SENSOR_TO_STATE, SensorArray
from rumen_twin.state_estimator import RumenStateEstimator


def _state(pH=6.40):
    return TwinState(pH, 38.9, 5.2, 0.24, 0.42, 0.02, 31.0, 0.88, 0.10, 0.05)


def test_noise_free_sensor_returns_the_truth(cfg):
    sa = SensorArray(cfg, np.random.default_rng(0),
                     {"pH": {"noise_sd": 0.0, "dropout_prob": 0.0}}, 10)
    assert sa.observe(_state(6.13), 0).get("pH") == pytest.approx(6.13)


def test_noise_is_unbiased_and_has_the_requested_sd(cfg):
    sd = 0.12
    sa = SensorArray(cfg, np.random.default_rng(1),
                     {"pH": {"noise_sd": sd, "dropout_prob": 0.0}}, 4000)
    v = np.array([sa.observe(_state(), t).get("pH") for t in range(4000)])
    assert abs(v.mean() - 6.40) < 0.01
    assert abs(v.std() - sd) < 0.012


def test_systematic_bias_shifts_the_reading(cfg):
    sa = SensorArray(cfg, np.random.default_rng(2),
                     {"pH": {"noise_sd": 0.0, "bias": -0.18, "dropout_prob": 0.0}}, 10)
    assert sa.observe(_state(6.40), 0).get("pH") == pytest.approx(6.22)


def test_dropout_probability_is_honoured(cfg):
    sa = SensorArray(cfg, np.random.default_rng(3), {"pH": {"dropout_prob": 0.30}}, 4000)
    miss = np.mean([sa.observe(_state(), t).get("pH") is None for t in range(4000)])
    assert 0.26 < miss < 0.34


def test_dropout_mask_blocks_exactly_the_masked_steps(cfg):
    mask = np.zeros(20, dtype=bool); mask[5:12] = True
    sa = SensorArray(cfg, np.random.default_rng(4),
                     {"pH": {"dropout_mask": mask, "dropout_prob": 0.0, "noise_sd": 0.0}}, 20)
    got = [sa.observe(_state(), t).get("pH") for t in range(20)]
    assert all(v is None for v in got[5:12])
    assert all(v is not None for v in got[:5] + got[12:])


@pytest.mark.parametrize("delay", [1, 2, 4])
def test_delayed_sensor_returns_a_stale_measurement(cfg, delay):
    """A delayed reading is genuinely old, not smoothed, and the first
    ``delay`` steps have nothing to report."""
    sa = SensorArray(cfg, np.random.default_rng(5),
                     {"pH": {"delay_steps": delay, "noise_sd": 0.0, "dropout_prob": 0.0}}, 30)
    truth, seen = [], []
    for t in range(20):
        s = _state(6.40 - 0.03 * t)
        truth.append(s.rumen_pH)
        seen.append(sa.observe(s, t).get("pH"))
    assert all(v is None for v in seen[:delay])
    for t in range(delay, 20):
        assert seen[t] == pytest.approx(truth[t - delay])


def test_latent_variables_have_no_sensor(cfg):
    sa = SensorArray(cfg, np.random.default_rng(6), {}, 10)
    obs = sa.observe(_state(), 0)
    for k in LATENT_VARIABLES:
        assert obs.get(k) is None and not obs.is_available(k)
        assert k not in SENSOR_TO_STATE


def test_observation_boost_reduces_noise_and_dropout(cfg):
    sa = SensorArray(cfg, np.random.default_rng(7),
                     {"pH": {"noise_sd": 0.2, "dropout_prob": 0.4}}, 3000)
    plain = [sa.observe(_state(), t, 1.0, 1.0).get("pH") for t in range(1500)]
    boost = [sa.observe(_state(), t, 0.3, 0.2).get("pH") for t in range(1500)]
    miss_plain = np.mean([v is None for v in plain])
    miss_boost = np.mean([v is None for v in boost])
    assert miss_boost < miss_plain
    assert np.std([v for v in boost if v is not None]) < \
           np.std([v for v in plain if v is not None])


# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["none", "hold", "ema", "kalman"])
def test_estimator_survives_a_fully_blind_sensor(cfg, method):
    """With no observations at all the estimate must stay finite and frozen."""
    sa = SensorArray(cfg, np.random.default_rng(8), {"pH": {"dropout_prob": 1.0}}, 40)
    est = RumenStateEstimator(cfg, method=method)
    est.reset({"pH": 6.4})
    for t in range(40):
        out = est.update(sa.observe(_state(5.6), t))
        assert np.isfinite(out.pH_estimate)
    assert out.pH_estimate == pytest.approx(6.4)
    assert out.steps_since_pH_obs == 40
    assert not out.pH_available


def test_kalman_uncertainty_grows_while_observations_are_missing(cfg):
    sa = SensorArray(cfg, np.random.default_rng(9), {"pH": {"dropout_prob": 1.0}}, 20)
    est = RumenStateEstimator(cfg, method="kalman")
    est.reset({"pH": 6.4})
    u = [est.update(sa.observe(_state(), t)).pH_uncertainty for t in range(20)]
    assert np.all(np.diff(u) > 0), "Kalman uncertainty must grow without measurements"


def test_estimator_beats_the_raw_sensor_under_noise(cfg):
    """Filtering must reduce the error relative to the raw noisy reading."""
    truth = 6.15
    sa = SensorArray(cfg, np.random.default_rng(10),
                     {"pH": {"noise_sd": 0.18, "dropout_prob": 0.0}}, 400)
    est = RumenStateEstimator(cfg, method="kalman")
    est.reset({"pH": truth})
    raw_err, est_err = [], []
    for t in range(400):
        o = sa.observe(_state(truth), t)
        raw_err.append(abs(o.get("pH") - truth))
        est_err.append(abs(est.update(o).pH_estimate - truth))
    assert np.mean(est_err[50:]) < np.mean(raw_err[50:])
