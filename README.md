# RumenTwin-RL

**Reinforcement-learning control of a synthetic digital twin of the dairy-cow rumen.**

> ### ⚠️ Synthetic simulation study
> This repository contains a **proof-of-concept simulation**. No real animal,
> sensor or experimental data is used anywhere. The rumen model is a
> transparent, deliberately simple dynamical system — it is **not a validated
> biochemical model**, the interventions are **not a veterinary treatment
> protocol**, and nothing produced here should inform the care of a real
> animal. Terms such as "SARA-like" denote *simulated model states*, not
> clinical diagnoses.

---

## Research questions

1. **Can a reinforcement-learning-controlled digital twin maintain rumen
   physiological stability under disturbances more effectively than static
   expert-derived rules?**
2. **How robust are reinforcement-learning policies when rumen sensor
   observations are incomplete, delayed or noisy?**

Both are answered inside the simulation only.

---

## Quick start

```bash
git clone <this-repository>
cd rumen-twin-rl

python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

python run_experiments.py        # trains the agents, then runs every experiment
python generate_results.py       # builds every table and figure from the saved episodes
```

Verify the whole pipeline in a few minutes before committing to the full run:

```bash
python run_experiments.py --quick
python generate_results.py
pytest -q
```

Interactive demo (secondary to the experiments):

```bash
streamlit run app/streamlit_app.py
```

No absolute paths are used anywhere; everything resolves relative to the
repository root.

---

## The closed loop

```
   cow physiological state s_t
        -> synthetic sensors          (noise, bias, dropout, delay)
        -> state estimation           (Kalman / EMA / hold)
        -> decision policy            (expert rules | DQN | PPO)
        -> intervention a_t
        -> rumen digital twin         s_{t+1} = f(s_t, a_t, e_t) + eps_t
        -> new observation
```

The controller **never** sees `s_t`. Three state variables — `vfa_index`,
`lactate_index` and `health_risk` — have **no sensor at all** and are
permanently latent, so the control problem is partially observable by
construction.

Time step: **15 minutes**. Default episode: **48 h** (192 steps).

---

## What is in the model

| Component | Where | Summary |
|---|---|---|
| State variables | `src/rumen_twin/physiology.py` | 10 core variables + 5 latent auxiliary pools, each with a unit, admissible range, default and process noise |
| Dynamics | `src/rumen_twin/physiology.py` | 12 documented difference equations; the pH balance is written in deviation form around a reference ration |
| Virtual population | `src/rumen_twin/cow.py` | 1000 cows, 9 correlated traits, hierarchical (population → cow → within-cow noise) |
| Disturbances | `src/rumen_twin/disturbances.py` | Scenarios A–J; feeding plan, heat load, sensor degradation |
| Sensors | `src/rumen_twin/sensors.py` | Gaussian noise, systematic bias with drift, dropout, transport delay |
| State estimator | `src/rumen_twin/state_estimator.py` | scalar Kalman filter (default), exponential smoothing, or hold-last |
| Interventions | `src/rumen_twin/interventions.py` | 7 discrete actions with effects, durations, costs, saturation and cooldowns |
| Reward | `src/rumen_twin/reward.py` | 8 itemised terms; 5 configurable weight sets |
| Expert baseline | `src/rumen_twin/expert_rules.py` | transparent threshold cascade, all thresholds in `config.yaml` |
| Environment | `src/rumen_twin/environment.py` | `RumenTwinEnv`, Gymnasium-compatible, 26-dim observation, 7 actions |
| Agents | `src/agents/` | DQN and PPO (Stable-Baselines3) |

### Scenarios

| | Scenario | Seen in RL training |
|---|---|---|
| A | Normal | yes |
| B | High concentrate | yes |
| C | Delayed feeding (with compensatory intake) | yes |
| D | Heat stress | yes |
| E | Gradual SARA-like disturbance | yes |
| F | pH sensor dropout | yes |
| G | pH sensor noise + drifting bias | yes |
| H | Sensor delay (15–60 min) | yes |
| I | Mixed: high concentrate + heat + noisy pH | **no — held out** |
| J | Stress test: heat + feed delay + sensor bias + delay | **no — held out** |

Scenarios F–H apply sensor degradation *on top of the same nutritional
disturbance as B*, so the clean-sensor reference for the robustness analysis
is scenario B and the comparison is like-for-like.

### Action space

| id | action | cost | duration | usage constraint |
|---|---|---|---|---|
| 0 | no intervention | 0.00 | — | — |
| 1 | increase forage ratio | 0.55 | 6 h | excess-use penalty |
| 2 | reduce concentrate | 0.85 | 4 h | excess-use penalty |
| 3 | administer buffer | 1.50 | 3 h | **saturating dose** |
| 4 | split feeding | 0.45 | 5 h | excess-use penalty |
| 5 | increase observation frequency | 0.25 | 3 h | excess-use penalty |
| 6 | trigger human/veterinary alert | 3.00 | 6 h | **cooldown**, false-alarm penalty |

Three independent mechanisms prevent a degenerate "always intervene" policy:
the per-action cost, a **quadratic** excess-intervention penalty over a rolling
6-h window, and an **exponentially saturating buffer dose** (the second dose
within 4 h delivers ~43 % of the first, the third ~18 %, at full price).

---

## Experimental protocol

* **Split by individual**: 700 training / 150 validation / 150 test cows.
  RL agents are trained **only** on training cows; checkpoints are selected on
  the **validation** cows; every reported number comes from the **test** cows,
  whose trait vectors the agents have never encountered.
* **180 episodes per (controller × scenario)** across 3 evaluation seeds —
  7 200 episodes in the main experiment alone.
* **Common random numbers**: every controller runs an identical
  `(cow, scenario, episode seed)` plan, and the cow at a given
  `(eval_seed, episode_index)` is the same in every scenario. Comparisons are
  therefore **paired** both across controllers and across scenarios.
* Results are written to `results/raw/*.parquet`; **nothing is entered by hand.**

### Statistics

Mean, SD, median, IQR and 95 % percentile-bootstrap CIs, with paired
comparisons reported as **Hedges' *g* (paired)** and the **matched-pairs
rank-biserial correlation**. Wilcoxon signed-rank p-values are reported but
deliberately de-emphasised: in a simulation the sample size is a budget
decision, so effect sizes and CIs carry the argument.

---

## Repository layout

```
rumen-twin-rl/
  config.yaml            every model, reward, training and evaluation parameter
  seeds.yaml             every random seed used anywhere in the pipeline
  run_experiments.py     orchestrator: train -> main -> ablation -> sensitivity
  generate_results.py    orchestrator: tables -> figures -> explainability
  src/rumen_twin/        the digital twin, environment, reward, rules, statistics
  src/agents/            DQN and PPO agents + controller wrappers
  scripts/               individual pipeline stages (each runnable on its own)
  tests/                 129 unit tests
  app/streamlit_app.py   interactive demo
  results/raw/           one row per simulated episode (generated)
  tables/  figures/      generated outputs (CSV + Markdown, PNG 300 dpi + PDF)
```

### Individual stages

```bash
python scripts/train_ppo.py --run ppo_main
python scripts/train_dqn.py --run dqn_main
python scripts/run_baselines.py               # no trained model needed
python scripts/run_experiments.py --skip-training
python scripts/ablation_study.py              # trains the 3 ablated agents
python scripts/sensitivity_analysis.py
python scripts/generate_tables.py
python scripts/generate_figures.py
python scripts/explainability.py
```

### A note on the manuscript

The manuscript built from these results is **not part of this repository** while
it is under submission. Everything needed to reproduce the numbers it reports is
here: the code, the configuration, the seeds and the raw per-episode records.

Figures are generated twice: `figures/` keeps standalone versions (headline
title and disclaimer baked in, so each plot is self-describing) and
`figures/manuscript/` versions intended for a document, where the caption would
carry that information instead.

---

## Reproducibility

* Every stochastic component draws from a seed declared in `seeds.yaml`;
  no unseeded RNG is used anywhere in the pipeline.
* Episode seeds come from a BLAKE2b-based hash, not Python's per-process-salted
  `hash()`, so plans are identical across runs and machines.
* Each episode reset spawns **three independent RNG streams** (scenario,
  physiology, sensors) from the episode seed, so the cow and the disturbance
  realisation do not depend on which controller is acting.
* `results/run_metadata.json` records the full config, seeds, library versions
  and platform for each run.
* `pytest -q` runs 129 tests covering physiological bounds, environment
  reset/step, reproducibility, disturbance generation, intervention effects,
  sensor dropout/delay, reward behaviour and the absence of NaN or
  physiologically impossible values.

## Outputs

| Table | Content |
|---|---|
| 1 | Digital-twin state variables |
| 2 | Simulation parameters |
| 3 | Intervention definitions |
| 4 | Experimental scenarios |
| 5 | Overall controller performance |
| 6 | Performance by scenario |
| 7 | Sensor-failure robustness |
| 8 | Ablation results |
| S1–S4 | Paired comparisons, reward sensitivity, decision timeline, policy sensitivity |

| Figure | Content |
|---|---|
| 1 | Architecture of RumenTwin-RL |
| 2 | Example 48-h trajectory with interventions |
| 3 | Low-pH exposure by controller |
| 4 | Performance by disturbance scenario |
| 5 | Robustness to sensor noise / dropout / delay |
| 6 | Intervention cost vs physiological stability |
| 7 | Individual variability across virtual cows |
| 8 | Generalisation to unseen mixed disturbances |
| 9–10 | Decision timeline and local policy sensitivity (explainability) |

Figures use a categorical palette validated for colour-vision deficiency, and
every multi-series figure carries a second, non-colour channel (marker shape,
line style or hatch) so identity never rests on hue alone.

## Licence and citation

Research code released for reproducibility. A manuscript describing this work is
in preparation; until it appears, please cite this repository and keep the
synthetic-simulation disclaimer intact.
