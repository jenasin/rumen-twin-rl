"""Gymnasium API conformance, info contract, determinism and safety."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rumen_twin.environment import OBS_DIM, OBS_NAMES, RumenTwinEnv
from rumen_twin.interventions import N_ACTIONS

REQUIRED_INFO_KEYS = ["cow_id", "scenario", "true_state", "observed_state", "pH",
                      "health_risk", "intervention_cost", "cumulative_cost",
                      "low_pH_duration", "action"]


def test_passes_gymnasium_env_checker(env):
    check_env(env, skip_render_check=True)


def test_spaces(env):
    assert env.action_space.n == N_ACTIONS
    assert env.observation_space.shape == (OBS_DIM,)
    assert len(OBS_NAMES) == OBS_DIM


def test_reset_returns_valid_observation_and_info(env):
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert np.isfinite(obs).all()
    for k in REQUIRED_INFO_KEYS:
        assert k in info, f"info is missing required key '{k}'"
    assert info["cow_split"] == "test"
    assert env.t == 0


def test_reset_restores_initial_conditions(env):
    env.reset(seed=5)
    for _ in range(50):
        env.step(3)
    assert env.manager.n_interventions > 0
    env.reset(seed=5)
    assert env.t == 0
    assert env.manager.n_interventions == 0
    assert env.manager.cumulative_cost == 0.0
    assert env.cumulative_reward == 0.0
    assert env.low_pH_steps == 0


@pytest.mark.parametrize("action", list(range(N_ACTIONS)))
def test_step_is_valid_for_every_action(env, action):
    env.reset(seed=2)
    obs, r, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert np.isfinite(r)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert info["action"] == action
    assert info["intervention_cost"] >= 0.0


def test_episode_runs_to_the_configured_horizon(env, cfg):
    expected = int(cfg["simulation"]["episode_hours"] * 60 / cfg["simulation"]["dt_minutes"])
    env.reset(seed=3)
    n, done = 0, False
    while not done:
        _, _, te, tr, _ = env.step(0)
        n += 1
        done = te or tr
    assert n == expected
    assert not te, "the environment must never terminate early (that would be rewardable)"


def test_invalid_action_raises(env):
    env.reset(seed=4)
    with pytest.raises(ValueError):
        env.step(N_ACTIONS + 3)


def test_observations_never_contain_nan_under_random_policy(env):
    r = np.random.default_rng(0)
    for scen in env.cfg["evaluation_scenarios"]:
        obs, _ = env.reset(seed=int(r.integers(1, 10 ** 6)), options={"scenario": scen})
        done = False
        while not done:
            obs, rew, te, tr, info = env.step(int(r.integers(N_ACTIONS)))
            assert np.isfinite(obs).all(), f"NaN/inf in observation ({scen})"
            assert np.isfinite(rew), f"NaN/inf reward ({scen})"
            assert np.isfinite(list(info["true_state"].values())).all()
            done = te or tr


def test_same_seed_gives_identical_episodes(env):
    def rollout(seed):
        obs, _ = env.reset(seed=seed, options={"scenario": "B_high_concentrate"})
        out = [obs.copy()]
        for a in [0, 1, 0, 3, 0, 2, 0, 0, 4, 0] * 5:
            obs, r, te, tr, info = env.step(a)
            out.append(obs.copy())
        return np.asarray(out), info["pH"], info["cumulative_reward"]

    a = rollout(777)
    b = rollout(777)
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]


def test_different_seeds_give_different_episodes(env):
    o1, i1 = env.reset(seed=1, options={"scenario": "B_high_concentrate"})
    o2, i2 = env.reset(seed=2, options={"scenario": "B_high_concentrate"})
    assert not np.array_equal(o1, o2) or i1["cow_id"] != i2["cow_id"]


def test_common_random_numbers_across_controllers(env):
    """The cow and the disturbance realisation must not depend on the actions.

    This is what makes the controller comparison paired.
    """
    def rollout(actions):
        env.reset(seed=4242, options={"scenario": "E_gradual_sara"})
        cow = env.cow.cow_id
        plan = env.scenario.planned_dmi.copy()
        heat = env.scenario.heat.copy()
        for a in actions:
            env.step(a)
        return cow, plan, heat

    c1, p1, h1 = rollout([0] * 40)
    c2, p2, h2 = rollout([3, 1, 2, 0, 4] * 8)
    assert c1 == c2
    assert np.array_equal(p1, p2)
    assert np.array_equal(h1, h2)


def test_rl_agent_cannot_see_latent_variables(env):
    """VFA, lactate and health risk must never reach the observation vector."""
    env.reset(seed=9)
    for latent in ("vfa_index", "lactate_index", "health_risk"):
        assert not any(latent in n for n in OBS_NAMES)
        assert env._last_obs_raw.get(latent) is None
        assert not env._last_obs_raw.is_available(latent)


def test_render_returns_text(env):
    env.reset(seed=11)
    env.step(0)
    txt = env.render()
    assert isinstance(txt, str) and "SYNTHETIC SIMULATION" in txt


def test_cow_split_is_respected(cfg, seeds, population):
    for split in ("train", "validation", "test"):
        e = RumenTwinEnv(config=cfg, seeds=seeds, population=population, split=split)
        for s in range(6):
            _, info = e.reset(seed=100 + s)
            assert info["cow_split"] == split
