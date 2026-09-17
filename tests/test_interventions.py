"""Intervention effects, durations, costs, saturation and cooldowns."""
import pytest

from rumen_twin.interventions import (ACTION_NAMES, N_ACTIONS, WASTED_CALL_COST,
                                      InterventionManager, intervention_table)


@pytest.fixture
def mgr(cfg):
    return InterventionManager(cfg, 192)


def test_no_intervention_is_free_and_changes_nothing(mgr):
    r = mgr.apply(0, 0)
    assert r.cost == 0.0 and not r.is_intervention
    m = mgr.modifiers(0, 0.60, 0.25)
    assert m.forage_ratio == pytest.approx(0.60)
    assert m.concentrate_scale == 1.0
    assert m.planned_dmi == pytest.approx(0.25)
    assert m.buffer_dose == 0.0
    assert mgr.cumulative_cost == 0.0


def test_increase_forage_raises_ratio_for_its_declared_duration(mgr, cfg):
    d = int(cfg["interventions"][1]["duration_steps"])
    delta = float(cfg["interventions"][1]["forage_delta"])
    mgr.apply(1, 0)
    assert mgr.modifiers(0, 0.60, 0.2).forage_ratio == pytest.approx(0.60 + delta)
    assert mgr.modifiers(d - 1, 0.60, 0.2).forage_ratio == pytest.approx(0.60 + delta)
    assert mgr.modifiers(d, 0.60, 0.2).forage_ratio == pytest.approx(0.60)


def test_forage_ratio_is_clamped_to_the_feasible_range(mgr, cfg):
    mgr.apply(1, 0)
    assert mgr.modifiers(0, 0.87, 0.2).forage_ratio <= cfg["feeding"]["forage_ratio_max"]


def test_reduce_concentrate_scales_the_concentrate_fraction(mgr, cfg):
    scale = float(cfg["interventions"][2]["concentrate_scale"])
    mgr.apply(2, 0)
    assert mgr.modifiers(0, 0.60, 0.2).concentrate_scale == pytest.approx(scale)
    assert mgr.modifiers(99, 0.60, 0.2).concentrate_scale == 1.0


def test_buffer_dose_saturates_with_repeated_use(mgr, cfg):
    """The anti-degeneracy mechanism: repeated buffering loses most of its effect."""
    doses = []
    for t in (0, 2, 4, 6):
        mgr.apply(3, t)
        doses.append(mgr.modifiers(t, 0.6, 0.2).buffer_dose)
    assert doses == sorted(doses, reverse=True)
    assert doses[1] < 0.5 * doses[0]
    assert doses[3] < 0.1 * doses[0]
    # but the cost is charged in full every time
    assert mgr.cumulative_cost == pytest.approx(4 * cfg["interventions"][3]["cost"])


def test_buffer_effect_recovers_after_the_saturation_window(cfg):
    m = InterventionManager(cfg, 400)
    m.apply(3, 0)
    first = m.modifiers(0, 0.6, 0.2).buffer_dose
    w = int(cfg["interventions"][3]["saturation_window"])
    m.apply(3, w + 5)
    later = m.modifiers(w + 5, 0.6, 0.2).buffer_dose
    assert later == pytest.approx(first)


def test_split_feeding_conserves_total_offered_mass(mgr):
    mgr.apply(4, 0)
    offered = [0.5, 0.5, 0.5] + [0.0] * 27
    delivered = sum(mgr.modifiers(t, 0.6, offered[t]).planned_dmi for t in range(30))
    assert delivered == pytest.approx(sum(offered), rel=1e-9)


def test_split_feeding_flattens_the_meal_peak(mgr):
    peak_plain = InterventionManager(mgr.cfg, 192).modifiers(0, 0.6, 0.9).planned_dmi
    mgr.apply(4, 0)
    assert mgr.modifiers(0, 0.6, 0.9).planned_dmi < peak_plain


def test_observation_boost_scales_noise_and_dropout(mgr, cfg):
    mgr.apply(5, 0)
    m = mgr.modifiers(0, 0.6, 0.2)
    assert m.noise_scale == pytest.approx(cfg["interventions"][5]["noise_scale"])
    assert m.dropout_scale == pytest.approx(cfg["interventions"][5]["dropout_scale"])
    assert mgr.modifiers(99, 0.6, 0.2).noise_scale == 1.0


def test_alert_cooldown_rejects_but_still_charges(mgr, cfg):
    cd = int(cfg["interventions"][6]["cooldown_steps"])
    assert mgr.apply(6, 0).accepted
    r = mgr.apply(6, 3)
    assert not r.accepted
    assert r.cost == WASTED_CALL_COST
    # a rejected call must still count, otherwise it is a free no-op to spam
    assert r.is_intervention
    assert mgr.n_interventions == 2
    assert mgr.apply(6, cd + 1).accepted
    assert mgr.n_alerts == 2


def test_alert_delivers_relief_and_support(mgr, cfg):
    mgr.apply(6, 0)
    m = mgr.modifiers(0, 0.60, 0.2)
    assert m.health_relief == pytest.approx(cfg["interventions"][6]["health_relief"])
    assert m.buffer_dose == pytest.approx(cfg["interventions"][6]["buffer_dose"])
    assert m.forage_ratio > 0.60


def test_costs_accumulate_and_match_the_configuration(mgr, cfg):
    total = 0.0
    for a in range(1, 6):
        total += float(cfg["interventions"][a]["cost"])
        mgr.apply(a, a * 30)
    assert mgr.cumulative_cost == pytest.approx(total)
    assert mgr.n_interventions == 5


def test_recent_intervention_window(mgr):
    """The rolling window counts interventions in (t-window, t]."""
    for t in (0, 5, 10, 40):
        mgr.apply(1, t)
    assert mgr.recent_intervention_count(10, 24) == 3     # t = 0, 5, 10
    assert mgr.recent_intervention_count(40, 24) == 1     # only t = 40 (17..40)
    assert mgr.recent_intervention_count(40, 48) == 4     # all four


def test_invalid_action_raises(mgr):
    with pytest.raises(ValueError):
        mgr.apply(N_ACTIONS, 0)
    with pytest.raises(ValueError):
        mgr.apply(-1, 0)


def test_reset_clears_all_state(mgr):
    for a in range(1, 7):
        mgr.apply(a, 0)
    mgr.reset()
    assert mgr.cumulative_cost == 0.0 and mgr.n_interventions == 0 and mgr.n_alerts == 0
    assert mgr.modifiers(0, 0.6, 0.25).forage_ratio == pytest.approx(0.6)
    assert not any(mgr.is_active(a, 0) for a in range(7))


def test_intervention_table_covers_every_action(cfg):
    t = intervention_table(cfg)
    assert len(t) == N_ACTIONS
    assert list(t["name"]) == ACTION_NAMES
