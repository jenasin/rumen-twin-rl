"""Physiological plausibility and numerical safety of the twin dynamics."""
import numpy as np
import pytest

from rumen_twin.disturbances import build_scenario
from rumen_twin.physiology import STATE_FIELDS, RumenPhysiology, StepInputs


def _run(cow, cfg, scenario, seed=0, n=192):
    rng = np.random.default_rng(seed)
    sc = build_scenario(scenario, cow, rng, n, cfg)
    phys = RumenPhysiology(cow, cfg, rng)
    phys.reset(rng)
    states = []
    for t in range(n):
        states.append(phys.step(StepInputs(
            planned_dmi=float(sc.planned_dmi[t]),
            forage_ratio=float(sc.forage_plan[t]),
            heat_load=float(sc.heat[t]))))
    return states


@pytest.mark.parametrize("scenario", ["A_normal", "B_high_concentrate", "D_heat_stress",
                                      "E_gradual_sara", "J_stress_test"])
def test_state_stays_within_admissible_bounds(cfg, population, scenario):
    """No state variable may ever leave its declared admissible range."""
    for cow in population.subset("train")[:12]:
        for s in _run(cow, cfg, scenario, seed=cow.cow_id):
            for f in STATE_FIELDS:
                v = getattr(s, f)
                meta = cfg["state_variables"][f]
                assert np.isfinite(v), f"{f} became non-finite in {scenario}"
                assert meta["min"] - 1e-9 <= v <= meta["max"] + 1e-9, \
                    f"{f}={v} outside [{meta['min']}, {meta['max']}] in {scenario}"


def test_no_nan_in_auxiliary_pools(cfg, population):
    for s in _run(population.subset("train")[0], cfg, "E_gradual_sara"):
        for f in ("concentrate_pool", "buffer_pool", "saliva_buffer", "forage_ratio",
                  "intake_ema"):
            assert np.isfinite(getattr(s, f))
        assert 0.0 <= s.concentrate_pool <= 1.0
        assert s.buffer_pool >= 0.0


def test_undisturbed_cow_is_physiologically_plausible(cfg, population):
    """Scenario A must produce a herd that is mostly healthy."""
    means = []
    for cow in population.subset("train")[:30]:
        st = _run(cow, cfg, "A_normal", seed=cow.cow_id)
        pH = np.array([s.rumen_pH for s in st])
        means.append((pH.mean(),
                      np.mean([s.rumination_minutes for s in st]) * cfg["simulation"]["steps_per_day"],
                      np.mean([s.dry_matter_intake for s in st]) * cfg["simulation"]["steps_per_day"],
                      np.mean([s.rumen_temperature for s in st]),
                      st[-1].health_risk))
    pH, rum, dmi, temp, hr = np.array(means).mean(axis=0)
    assert 6.25 <= pH <= 6.60, f"baseline mean pH implausible: {pH}"
    assert 380 <= rum <= 620, f"baseline rumination implausible: {rum} min/day"
    assert 16 <= dmi <= 28, f"baseline DMI implausible: {dmi} kg/day"
    assert 38.4 <= temp <= 39.6, f"baseline temperature implausible: {temp}"
    assert hr < 0.15, f"undisturbed cows should not accumulate health risk: {hr}"


def test_state_has_inertia(cfg, population):
    """pH must not jump discontinuously between consecutive 15-min steps."""
    for cow in population.subset("train")[:10]:
        st = _run(cow, cfg, "B_high_concentrate", seed=cow.cow_id)
        d = np.abs(np.diff([s.rumen_pH for s in st]))
        assert d.max() < 0.35, f"pH changed by {d.max():.3f} in one step (no inertia)"


def test_high_concentrate_lowers_pH_relative_to_normal(cfg, population):
    """Mechanism 1/2: more rapidly fermentable carbohydrate depresses pH."""
    normal, high = [], []
    for cow in population.subset("train")[:20]:
        normal.append(np.mean([s.rumen_pH for s in _run(cow, cfg, "A_normal", cow.cow_id)]))
        high.append(np.mean([s.rumen_pH for s in _run(cow, cfg, "B_high_concentrate", cow.cow_id)]))
    assert np.mean(high) < np.mean(normal) - 0.15


def test_buffer_raises_pH(cfg, population):
    """Mechanism 3: administered buffer increases pH, all else equal."""
    cow = population.subset("train")[0]
    out = {}
    for dose in (0.0, 0.55):
        rng = np.random.default_rng(7)
        sc = build_scenario("B_high_concentrate", cow, rng, 96, cfg)
        ph = RumenPhysiology(cow, cfg, np.random.default_rng(7)); ph.reset(np.random.default_rng(7))
        vals = []
        for t in range(96):
            ph.step(StepInputs(planned_dmi=float(sc.planned_dmi[t]),
                               forage_ratio=float(sc.forage_plan[t]),
                               buffer_dose=dose if t in (40, 52, 64) else 0.0,
                               heat_load=float(sc.heat[t])))
            vals.append(ph.state.rumen_pH)
        out[dose] = np.mean(vals[40:80])
    assert out[0.55] > out[0.0], "buffer administration did not raise pH"


def test_heat_stress_reduces_intake_and_rumination(cfg, population):
    """Mechanism 7: heat load depresses DMI and rumination."""
    base, heat = [], []
    for cow in population.subset("train")[:15]:
        b = _run(cow, cfg, "A_normal", cow.cow_id)
        h = _run(cow, cfg, "D_heat_stress", cow.cow_id)
        base.append((np.mean([s.dry_matter_intake for s in b]),
                     np.mean([s.rumination_minutes for s in b])))
        heat.append((np.mean([s.dry_matter_intake for s in h]),
                     np.mean([s.rumination_minutes for s in h])))
    b, h = np.array(base).mean(0), np.array(heat).mean(0)
    assert h[0] < b[0], "heat stress did not reduce dry matter intake"
    assert h[1] < b[1], "heat stress did not reduce rumination"


def test_sustained_low_pH_raises_health_risk(cfg, population):
    """Mechanism 8: prolonged low pH must accumulate health risk."""
    risks = [_run(c, cfg, "E_gradual_sara", c.cow_id)[-1].health_risk
             for c in population.subset("train")[:20]]
    assert np.mean(risks) > 0.10, "gradual SARA-like load produced no health risk"


def test_more_forage_supports_higher_pH(cfg, population):
    """Mechanism 2/6: a higher forage ratio raises rumination and pH."""
    cow = population.subset("train")[0]
    res = {}
    for phi in (0.40, 0.75):
        rng = np.random.default_rng(3)
        sc = build_scenario("A_normal", cow, rng, 192, cfg)
        ph = RumenPhysiology(cow, cfg, np.random.default_rng(3)); ph.reset(np.random.default_rng(3))
        pH, rum = [], []
        for t in range(192):
            ph.step(StepInputs(planned_dmi=float(sc.planned_dmi[t]), forage_ratio=phi,
                               heat_load=float(sc.heat[t])))
            pH.append(ph.state.rumen_pH); rum.append(ph.state.rumination_minutes)
        res[phi] = (np.mean(pH[96:]), np.mean(rum[96:]))
    assert res[0.75][0] > res[0.40][0], "more forage did not raise pH"
    assert res[0.75][1] > res[0.40][1], "more forage did not raise rumination"
