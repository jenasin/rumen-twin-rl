#!/usr/bin/env python3
"""RumenTwin-RL — run the full experimental protocol.

    python run_experiments.py                # everything (trains agents if needed)
    python run_experiments.py --stage main   # just the main controller comparison
    python run_experiments.py --quick        # small budgets, for a smoke test

This is a thin orchestrator over ``scripts/``; each stage can also be run on
its own. Afterwards run ``python generate_results.py`` to build every table and
figure from the saved episode records.

SYNTHETIC SIMULATION STUDY - not a veterinary decision tool.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

STAGES = {
    "train": [("train_ppo.py", ["--run", "ppo_main"]), ("train_dqn.py", ["--run", "dqn_main"])],
    "main": [("run_experiments.py", ["--skip-training"])],
    "ablation": [("ablation_study.py", [])],
    "sensitivity": [("sensitivity_analysis.py", [])],
}
QUICK_ARGS = {
    "train_ppo.py": ["--timesteps", "60000"],
    "train_dqn.py": ["--timesteps", "40000"],
    "run_experiments.py": ["--episodes", "15"],
    "ablation_study.py": ["--episodes", "15", "--timesteps", "60000"],
    "sensitivity_analysis.py": ["--episodes", "10"],
}


def run(script: str, args, quick: bool) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    if quick:
        cmd += QUICK_ARGS.get(script, [])
    print(f"\n$ {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise SystemExit(f"Stage failed: {script} (exit {res.returncode})")
    print(f"  [{script} finished in {time.time() - t0:.1f}s]", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RumenTwin-RL experiments")
    ap.add_argument("--stage", nargs="*", default=["train", "main", "ablation", "sensitivity"],
                    choices=list(STAGES))
    ap.add_argument("--quick", action="store_true",
                    help="tiny budgets: verifies the pipeline end to end in a few minutes")
    args = ap.parse_args()

    print("=" * 74)
    print("RumenTwin-RL — synthetic simulation study")
    print("Synthetic research simulation — not a veterinary decision tool.")
    print("=" * 74)
    t0 = time.time()
    for stage in args.stage:
        print(f"\n{'=' * 74}\nSTAGE: {stage}\n{'=' * 74}")
        for script, extra in STAGES[stage]:
            run(script, extra, args.quick)
    print(f"\nAll requested stages complete in {time.time() - t0:.1f}s.")
    print("Next: python generate_results.py")


if __name__ == "__main__":
    main()
