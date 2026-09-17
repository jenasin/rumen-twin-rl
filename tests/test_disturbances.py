"""Scenario generation: each disturbance must actually do what it claims."""
import numpy as np
import pytest

from rumen_twin.disturbances import build_scenario, scenario_names


@pytest.fixture
def cow(population):
    return population.subset("train")[0]


def _s(name, cow, cfg, seed=0, n=192):
    return build_scenario(name, cow, np.random.default_rng(seed), n, cfg)


@pytest.mark.parametrize("name", ["A_normal", "B_high_concentrate", "C_delayed_feeding",
                                  "D_heat_stress", "E_gradual_sara", "F_sensor_dropout",
                                  "G_sensor_noise", "H_sensor_delay", "I_mixed",
                                  "J_stress_test"])
def test_every_scenario_is_well_formed(name, cow, cfg):
    s = _s(name, cow, cfg)
    assert len(s) == 192
    for arr in (s.planned_dmi, s.forage_plan, s.heat):
        assert arr.shape == (192,)
        assert np.isfinite(arr).all()
    assert (s.planned_dmi >= 0).all()
    assert (s.forage_plan >= cfg["feeding"]["forage_ratio_min"]).all()
    assert (s.forage_plan <= cfg["feeding"]["forage_ratio_max"]).all()
    assert ((s.heat >= 0) & (s.heat <= 1)).all()
    assert s.label and s.family


def test_unknown_scenario_raises(cow, cfg):
    with pytest.raises(KeyError):
        _s("Z_does_not_exist", cow, cfg)


def test_scenario_generation_is_reproducible(cow, cfg):
    a, b = _s("J_stress_test", cow, cfg, seed=5), _s("J_stress_test", cow, cfg, seed=5)
    assert np.array_equal(a.planned_dmi, b.planned_dmi)
    assert np.array_equal(a.forage_plan, b.forage_plan)
    assert np.array_equal(a.heat, b.heat)
    c = _s("J_stress_test", cow, cfg, seed=6)
    assert not np.array_equal(a.planned_dmi, c.planned_dmi)


def test_normal_scenario_offers_roughly_the_cow_s_intake_capacity(cow, cfg):
    s = _s("A_normal", cow, cfg)
    per_day = s.planned_dmi.sum() / (192 / cfg["simulation"]["steps_per_day"])
    assert 0.8 * cow.intake_capacity_kg_day < per_day < 1.2 * cow.intake_capacity_kg_day


def test_feeding_plan_is_meal_structured(cow, cfg):
    """Intake must be concentrated into meals, not spread uniformly."""
    s = _s("A_normal", cow, cfg)
    day = s.planned_dmi[:96]
    assert day.max() > 3.0 * day.mean(), "feeding plan has no meal peaks"
    top_quarter = np.sort(day)[-24:].sum() / day.sum()
    assert top_quarter > 0.70, "meals should carry most of the daily intake"
    n_meals = len(cfg["feeding"]["meal_hours"])
    peaks = np.flatnonzero((day[1:-1] > day[:-2]) & (day[1:-1] > day[2:])
                           & (day[1:-1] > day.mean()))
    assert len(peaks) == n_meals, f"expected {n_meals} meal peaks, found {len(peaks)}"


def test_high_concentrate_reduces_the_forage_ratio(cow, cfg):
    s = _s("B_high_concentrate", cow, cfg)
    target = cfg["scenarios"]["B_high_concentrate"]["forage_ratio"]
    assert s.forage_plan[0] == pytest.approx(cfg["feeding"]["baseline_forage_ratio"])
    assert s.forage_plan[-1] == pytest.approx(target, abs=1e-6)
    assert np.all(np.diff(s.forage_plan) <= 1e-9), "forage ratio should ramp down monotonically"


def test_gradual_sara_ramps_the_acidogenic_load(cow, cfg):
    s = _s("E_gradual_sara", cow, cfg)
    assert s.forage_plan[0] > s.forage_plan[-1]
    assert s.planned_dmi.sum() > _s("A_normal", cow, cfg).planned_dmi.sum()


def test_heat_stress_is_diurnal_and_reaches_the_configured_peak(cow, cfg):
    s = _s("D_heat_stress", cow, cfg)
    peak = cfg["scenarios"]["D_heat_stress"]["peak_heat"]
    assert s.heat.max() > 0.8 * peak
    assert s.heat.min() < 0.3 * peak                      # cool at night
    assert _s("A_normal", cow, cfg).heat.max() < s.heat.max()


def test_delayed_feeding_shifts_and_sharpens_a_meal(cow, cfg):
    normal = _s("C_delayed_feeding", cow, cfg).planned_dmi[:96]
    base = _s("A_normal", cow, cfg).planned_dmi[:96]
    assert normal.max() > base.max(), "rebound feeding should raise the peak intake rate"
    assert not np.allclose(normal, base)


def test_sensor_scenarios_only_touch_the_observation_channel(cow, cfg):
    base = _s("B_high_concentrate", cow, cfg, seed=3)
    for name in ("F_sensor_dropout", "G_sensor_noise", "H_sensor_delay"):
        s = _s(name, cow, cfg, seed=3)
        assert np.allclose(s.forage_plan, base.forage_plan), \
            f"{name} must not change the nutritional disturbance"
        assert s.sensor_overrides, f"{name} declared no sensor override"


def test_dropout_scenario_produces_blocks_of_missing_data(cow, cfg):
    s = _s("F_sensor_dropout", cow, cfg, seed=1)
    mask = np.asarray(s.sensor_overrides["pH"]["dropout_mask"])
    assert mask.dtype == bool and mask.shape == (192,)
    assert 0.0 < mask.mean() < 0.9
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0])))[::2]
    assert runs.max() >= cfg["scenarios"]["F_sensor_dropout"]["block_len_steps"][0]


def test_noise_scenario_degrades_the_pH_sensor(cow, cfg):
    o = _s("G_sensor_noise", cow, cfg).sensor_overrides["pH"]
    assert o["noise_sd"] > cfg["sensors"]["pH"]["noise_sd"]
    assert o["bias"] != 0.0


def test_delay_scenario_is_within_15_to_60_minutes(cow, cfg):
    dt = cfg["simulation"]["dt_minutes"]
    for seed in range(12):
        d = _s("H_sensor_delay", cow, cfg, seed=seed).sensor_overrides["pH"]["delay_steps"]
        assert 15 <= d * dt <= 60


def test_held_out_scenarios_are_excluded_from_training(cfg):
    train = set(cfg["training_scenarios"])
    assert "I_mixed" not in train
    assert "J_stress_test" not in train
    assert set(cfg["evaluation_scenarios"]) >= train


def test_mixed_and_stress_scenarios_combine_several_disturbances(cow, cfg):
    for name in ("I_mixed", "J_stress_test"):
        s = _s(name, cow, cfg)
        assert s.forage_plan.min() < cfg["feeding"]["baseline_forage_ratio"]  # nutritional
        assert s.heat.max() > 0.5                                            # environmental
        assert "pH" in s.sensor_overrides                                    # sensor


def test_scenario_name_helpers(cfg):
    assert set(scenario_names(cfg, "training")) == set(cfg["training_scenarios"])
    assert set(scenario_names(cfg, "evaluation")) == set(cfg["evaluation_scenarios"])
