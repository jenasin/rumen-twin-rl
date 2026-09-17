#!/usr/bin/env python3
"""RumenTwin-RL — build every table and figure from the saved episode records.

    python generate_results.py

Reads only ``results/raw/*`` and ``config.yaml``; writes ``tables/`` and
``figures/``. No value is ever entered by hand.

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

# (script, args, description, required input relative to the repository root)
# A step whose required input is absent is skipped with a note rather than
# failing: the manuscript sources are not part of the public repository while
# the paper is under submission, so the tables and figures must still build.
STEPS = [
    ("generate_tables.py", [], "Tables 1-8 and supplementary tables", None),
    ("generate_figures.py", [], "Figures 1-8", None),
    ("explainability.py", [], "Figures 9-10 and the decision timeline", None),
    ("build_paper.py", [], "manuscript markdown, every number substituted",
     "paper/paper_template.md"),
    ("generate_figures.py", ["--manuscript"], "Figures 1-8 without in-image titles", None),
    ("explainability.py", ["--manuscript", "--reuse"],
     "Figures 9-10 without in-image titles", None),
    ("build_manuscript.py", [], "manuscript DOCX + PDF with figures embedded",
     "paper/paper_draft.md"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate all RumenTwin-RL results")
    ap.add_argument("--skip", nargs="*", default=[], help="script names to skip")
    args = ap.parse_args()

    if not any((ROOT / "results" / "raw").glob("main_results.*")):
        raise SystemExit("No results found in results/raw/. Run `python run_experiments.py` first.")

    print("=" * 74)
    print("RumenTwin-RL — generating results")
    print("Synthetic research simulation — not a veterinary decision tool.")
    print("=" * 74)
    t0 = time.time()
    skipped = []
    for script, extra, what, needs in STEPS:
        if script in args.skip:
            print(f"\n(skipping {script})")
            continue
        if needs and not (ROOT / needs).exists():
            print(f"\n(skipping {script}: {needs} not present)")
            skipped.append(script)
            continue
        print(f"\n$ {script}   # {what}", flush=True)
        res = subprocess.run([sys.executable, str(SCRIPTS / script), *extra], cwd=ROOT)
        if res.returncode != 0:
            raise SystemExit(f"Failed: {script} (exit {res.returncode})")
    print(f"\nDone in {time.time() - t0:.1f}s.  See tables/ and figures/.")
    if skipped:
        print("Manuscript steps were skipped (sources not present): "
              + ", ".join(sorted(set(skipped))))


if __name__ == "__main__":
    main()
