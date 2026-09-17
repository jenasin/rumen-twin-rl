#!/usr/bin/env python3
"""RumenTwin-RL — interactive digital-twin demo.

    streamlit run app/streamlit_app.py

Secondary to the experiments: this is a viewer for the same simulation code,
not a separate model. Everything it shows is generated live by
``rumen_twin.environment.RumenTwinEnv``.

SYNTHETIC RESEARCH SIMULATION — NOT A VETERINARY DECISION TOOL.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from rumen_twin.cow import CowPopulation
from rumen_twin.environment import RumenTwinEnv
from rumen_twin.expert_rules import ExpertRuleController, NoInterventionController
from rumen_twin.interventions import ACTION_NAMES
from rumen_twin.utils import load_config, load_seeds

st.set_page_config(page_title="RumenTwin-RL Digital Twin", page_icon="🐄", layout="wide")

CONTROLLER_LABEL = {"no_intervention": "No intervention", "expert_rules": "Expert rules",
                    "dqn": "DQN (reinforcement learning)", "ppo": "PPO (reinforcement learning)"}
COLOUR = {"no_intervention": "#4a3aa7", "expert_rules": "#2a78d6",
          "dqn": "#eb6834", "ppo": "#1baf7a"}


# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_static():
    cfg, sds = load_config(), load_seeds()
    return cfg, sds, CowPopulation(config=cfg, seeds=sds)


@st.cache_resource(show_spinner=False)
def _load_controller(name: str, _cfg):
    if name == "no_intervention":
        return NoInterventionController(_cfg), True
    if name == "expert_rules":
        return ExpertRuleController(_cfg), True
    from agents.common import load_controllers
    return load_controllers([name], _cfg)[0], False


def available_controllers(cfg):
    out = ["no_intervention", "expert_rules"]
    from agents.common import model_exists
    if model_exists("dqn_main"):
        out.append("dqn")
    if model_exists("ppo_main"):
        out.append("ppo")
    return out


# ---------------------------------------------------------------------------
st.title("🐄 RumenTwin-RL — Digital Twin of the Bovine Rumen")
st.error("**Synthetic research simulation — not a veterinary decision tool.** "
         "Every value on this page is produced by a proof-of-concept dynamical model. "
         "No real animal, sensor or experimental data is involved, and nothing here "
         "should inform the care of a real animal.", icon="⚠️")

cfg, sds, pop = _load_static()
test_cows = pop.subset("test")
thr = cfg["thresholds"]

# ------------------------------ sidebar ------------------------------------
with st.sidebar:
    st.header("Simulation setup")
    cow_id = st.selectbox("Cow ID", [c.cow_id for c in test_cows], index=0,
                          help="Held-out test cows only — these were never seen during RL training.")
    scenario = st.selectbox("Scenario", list(cfg["evaluation_scenarios"]),
                            format_func=lambda s: cfg["scenarios"][s]["label"], index=4)
    ctrls = available_controllers(cfg)
    controller = st.selectbox("Controller", ctrls,
                              format_func=lambda c: CONTROLLER_LABEL.get(c, c),
                              index=len(ctrls) - 1)
    noise_level = st.slider("Extra pH sensor noise (SD)", 0.0, 0.40,
                            float(cfg["sensors"]["pH"]["noise_sd"]), 0.01,
                            help="Overrides the pH sensor noise on top of the scenario.")
    dropout = st.slider("pH sensor dropout probability", 0.0, 0.9, 0.0, 0.05)
    seed = st.number_input("Random seed", 0, 10 ** 6, 2024, 1)
    speed = st.select_slider("Playback speed (steps per frame)", [1, 2, 4, 8, 16], value=4)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    start = c1.button("▶ Start", width="stretch")
    pause = c2.button("⏸ Pause", width="stretch")
    reset = c3.button("⟲ Reset", width="stretch")

    st.caption(f"{cfg['scenarios'][scenario]['description']}")
    st.caption(f"Model: {cfg['project']['name']} v{cfg['project']['version']} — "
               f"{cfg['project']['study_type']}")


# ------------------------------ episode state ------------------------------
def build_episode():
    env = RumenTwinEnv(config=cfg, seeds=sds, population=pop, split="test",
                       scenarios=[scenario], record_trajectory=True)
    overrides = {"pH": {"noise_sd": float(noise_level), "dropout_prob": float(dropout)}}
    obs, _ = env.reset(seed=int(seed), options={"cow_id": int(cow_id), "scenario": scenario,
                                                "sensor_overrides": overrides})
    ctrl, uses_ctx = _load_controller(controller, cfg)
    if hasattr(ctrl, "reset"):
        ctrl.reset()
    return {"env": env, "obs": obs, "ctrl": ctrl, "uses_ctx": uses_ctx,
            "rows": [], "done": False, "running": False,
            "key": (cow_id, scenario, controller, noise_level, dropout, seed)}


key = (cow_id, scenario, controller, noise_level, dropout, seed)
if "ep" not in st.session_state or st.session_state.ep["key"] != key or reset:
    st.session_state.ep = build_episode()
ep = st.session_state.ep

if start:
    ep["running"] = True
if pause:
    ep["running"] = False


def advance(n: int) -> None:
    env, ctrl = ep["env"], ep["ctrl"]
    for _ in range(n):
        if ep["done"]:
            return
        a = ctrl.act(env.controller_context()) if ep["uses_ctx"] else ctrl.act(ep["obs"])
        ep["obs"], r, te, tr, info = env.step(int(a))
        s = env.physio.state
        ep["rows"].append({
            "hour": (env.t - 1) * cfg["simulation"]["dt_minutes"] / 60.0,
            "true pH": s.rumen_pH,
            "observed pH": info["observed_pH"],
            "estimated pH": info["estimated_pH"],
            "temperature": s.rumen_temperature,
            "rumination": s.rumination_minutes * cfg["simulation"]["steps_per_day"],
            "VFA index": s.vfa_index,
            "lactate index": s.lactate_index,
            "health risk": s.health_risk,
            "milk proxy": s.milk_yield_proxy,
            "action": int(a),
            "action name": ACTION_NAMES[int(a)],
            "reward": r,
            "cumulative cost": info["cumulative_cost"],
        })
        ep["done"] = te or tr


if ep["running"] and not ep["done"]:
    advance(int(speed))

df = pd.DataFrame(ep["rows"])

# ------------------------------ current state ------------------------------
st.subheader("Current state of the twin")
if df.empty:
    st.info("Press **▶ Start** in the sidebar to run the simulation.")
    cur = None
else:
    cur = df.iloc[-1]
    k = st.columns(6)
    k[0].metric("Rumen pH", f"{cur['true pH']:.2f}",
                delta=None if len(df) < 2 else f"{cur['true pH'] - df.iloc[-2]['true pH']:+.3f}")
    k[1].metric("Temperature (°C)", f"{cur['temperature']:.2f}")
    k[2].metric("Rumination (min/d eq.)", f"{cur['rumination']:.0f}")
    k[3].metric("VFA index", f"{cur['VFA index']:.2f}", help="Latent — no sensor observes this.")
    k[4].metric("Milk proxy (kg/d)", f"{cur['milk proxy']:.1f}")
    k[5].metric("Cumulative cost", f"{cur['cumulative cost']:.2f}")

    risk = float(cur["health risk"])
    band = ("🟢 low" if risk < 0.15 else "🟡 elevated" if risk < 0.40 else
            "🟠 high" if risk < 0.70 else "🔴 very high")
    c1, c2, c3 = st.columns([2, 2, 3])
    c1.markdown(f"**Simulated health-risk indicator:** {band} ({risk:.2f})")
    c1.progress(min(1.0, risk))
    c2.markdown(f"**Chosen intervention:** `{cur['action name']}`")
    below = 100.0 * float((df["true pH"] < thr["low_pH"]).mean())
    c3.markdown(f"**Time below pH {thr['low_pH']:g} so far:** {below:.1f}%  \n"
                f"**Progress:** {len(df)}/{ep['env'].n_steps} steps "
                f"({df['hour'].iloc[-1]:.1f} h)")

# ------------------------------ timeline -----------------------------------
st.subheader("Timeline")
if not df.empty:
    tab1, tab2, tab3, tab4 = st.tabs(["Rumen pH", "Rumination & intake",
                                      "Latent state", "Interventions"])
    with tab1:
        d = df.set_index("hour")[["true pH", "observed pH", "estimated pH"]]
        st.line_chart(d, height=330,
                      color=[COLOUR[controller], "#9a9994", "#52514e"])
        st.caption(f"Dashed reference: low-pH threshold {thr['low_pH']:g}, "
                   f"severe threshold {thr['severe_low_pH']:g}. "
                   f"'observed' is what the sensor reported (it can be missing or biased); "
                   f"'estimated' is the controller's belief; 'true' is latent.")
    with tab2:
        st.line_chart(df.set_index("hour")[["rumination"]], height=250, color=["#2a78d6"])
        st.line_chart(df.set_index("hour")[["milk proxy"]], height=200, color=["#1baf7a"])
    with tab3:
        st.line_chart(df.set_index("hour")[["VFA index", "lactate index", "health risk"]],
                      height=300, color=["#52514e", "#9c2b2b", "#d03b3b"])
        st.caption("None of these variables has a sensor — no controller can observe them.")
    with tab4:
        acted = df[df["action"] > 0]
        if acted.empty:
            st.write("No interventions taken yet.")
        else:
            st.dataframe(acted[["hour", "action name", "reward", "cumulative cost"]]
                         .rename(columns={"hour": "time (h)"}),
                         width="stretch", hide_index=True, height=300)
            st.bar_chart(acted["action name"].value_counts(), height=220)

    st.download_button("Download this episode as CSV", df.to_csv(index=False).encode(),
                       file_name=f"rumentwin_cow{cow_id}_{scenario}_{controller}.csv",
                       mime="text/csv")

if ep["done"]:
    st.success(f"Episode complete ({ep['env'].n_steps} steps = "
               f"{cfg['simulation']['episode_hours']} simulated hours).")
elif ep["running"]:
    st.rerun()

st.markdown("---")
st.caption("RumenTwin-RL — synthetic simulation study. The interventions shown are "
           "simulated decision-support actions inside a proof-of-concept model; they do "
           "not represent a veterinary treatment protocol and are not validated against "
           "any real animal data.")
