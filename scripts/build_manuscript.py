#!/usr/bin/env python3
"""Build the conference-ready manuscript: DOCX and PDF with figures embedded.

Takes ``paper/paper_draft.md`` (already built from the results by
``build_paper.py``), inserts each figure immediately after the paragraph that
first refers to it, attaches a caption, and converts the result with pandoc.

    python scripts/build_manuscript.py                 # both formats
    python scripts/build_manuscript.py --formats docx  # just Word

Figures come from ``figures/manuscript/`` — the same plots without their
in-image headline, because the document caption carries it. Generate them with
``generate_figures.py --manuscript`` and ``explainability.py --manuscript``
(``generate_results.py`` does this automatically).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import _paths  # noqa: F401
from rumen_twin.utils import DISCLAIMER, PATHS, banner

FIG_DIR = PATHS["figures"] / "manuscript"

# figure number -> (file stem, caption)
FIGURES: Dict[int, tuple] = {
    1: ("figure1_architecture",
        "Architecture of RumenTwin-RL. The true physiological state is never "
        "shown to the controller: it is observed only through the synthetic "
        "sensor layer and the state estimator. Three state variables (VFA, "
        "lactate and the accumulated health-risk index) have no sensor at all."),
    2: ("figure2_example_trajectory",
        "Example 48-h trajectory for one held-out test cow under the gradual "
        "SARA-like disturbance, with and without control. Grey dots are the raw "
        "pH sensor readings; shaded bands mark the low-pH and severe-low-pH "
        "regions. The second panel shows every intervention the PPO policy "
        "ordered. The bottom panel shows the latent health-risk index, which no "
        "controller can observe. The illustrated cow was selected by a stated "
        "rule (the 75th percentile of low-pH exposure under no intervention), "
        "not by hand."),
    3: ("figure3_low_pH_exposure",
        "Low-pH exposure by controller, pooled over all 10 evaluation scenarios "
        "(1,800 episodes per controller). Bars are means with 95 % "
        "percentile-bootstrap confidence intervals; faint dots are individual "
        "episodes. Bars additionally carry a distinct hatch per controller."),
    4: ("figure4_performance_by_scenario",
        "Performance by disturbance scenario. Error bars are 95 % bootstrap "
        "confidence intervals. Shaded columns (I, J) are disturbance "
        "combinations held out of RL training and evaluated without retraining."),
    5: ("figure5_sensor_robustness",
        "Robustness to sensor degradation. All four conditions share the same "
        "underlying nutritional disturbance, so the change relative to the "
        "clean-sensor reference isolates the effect of the observation channel. "
        "Shaded ribbons are 95 % bootstrap confidence intervals. Note that the "
        "negatively biased sensor (noise + bias) *lowers* true low-pH exposure "
        "for every controller while raising its costs."),
    6: ("figure6_cost_vs_stability",
        "Intervention cost against physiological stability. Left: individual "
        "episodes (dots) with each controller's mean and 95 % confidence "
        "interval (large markers). Right: one point per (controller x scenario) "
        "condition, showing that the ordering is not driven by a single "
        "scenario."),
    7: ("figure7_individual_variability",
        "Individual variability across the 150 held-out test cows. Left: "
        "empirical cumulative distribution of per-cow mean low-pH exposure. "
        "Right: each point is one test cow, comparing a controller against the "
        "no-intervention baseline on the same animal; points below the diagonal "
        "improved."),
    8: ("figure8_generalisation_unseen",
        "Generalisation to disturbance combinations never seen during RL "
        "training (scenarios I and J), with no retraining. All controllers, "
        "including the rule-based one which cannot overfit, lose stability on "
        "the held-out combinations, indicating that those scenarios are "
        "intrinsically harder rather than that the learned policies failed to "
        "transfer."),
    9: ("figure9_explainability_timeline",
        "Decision timeline for one held-out test episode under PPO. Panels "
        "show, at every 15-minute step: the latent true pH against what the "
        "sensor reported and what the estimator believed; the action taken; "
        "rumination; the latent drivers no controller observes; and the reward "
        "earned. Vertical rules in the top panel mark the steps at which the "
        "controller intervened."),
    10: ("figure10_policy_sensitivity",
         "Local sensitivity of the PPO policy. Each observation feature was "
         "perturbed by +/- 1 SD across 900 states actually visited by the "
         "policy, holding the others fixed. The ranking, not the absolute "
         "level, is the informative part: the policy selects 'no intervention' "
         "in most states, so a one-SD nudge rarely moves it. This is a local "
         "diagnostic of decision sensitivity, not a causal attribution of "
         "importance."),
}

YAML_HEADER = """---
title: "RumenTwin-RL: Reinforcement Learning Control of a Synthetic Digital Twin of the Dairy Cow Rumen"
subtitle: "Synthetic simulation study"
date: "{date}"
abstract-title: ""
keywords: [digital twin, precision livestock farming, reinforcement learning, closed-loop control, partial observability, simulation study]
lang: en-GB
geometry: "a4paper,margin=2.2cm"
fontsize: 10pt
linestretch: 1.05
colorlinks: true
linkcolor: black
urlcolor: blue
header-includes:
  - \\usepackage{{etoolbox}}
  - \\AtBeginEnvironment{{longtable}}{{\\scriptsize}}
  - \\AtBeginEnvironment{{tabular}}{{\\scriptsize}}
  - \\usepackage{{float}}
  - \\usepackage{{caption}}
  - \\captionsetup[figure]{{font=small,labelformat=empty,skip=4pt}}
---

"""


def insert_figures(md: str, width_cm: float = 16.0) -> str:
    """Place each figure after the paragraph that first mentions it."""
    blocks = md.split("\n\n")
    placed: set = set()
    out: List[str] = []
    fig_ref = re.compile(r"Figure[s]?\s+(\d+)(?:\s*[-–]\s*(\d+))?")

    for block in blocks:
        out.append(block)
        if block.lstrip().startswith(("#",)):        # don't split a heading from its text
            continue
        mentioned: List[int] = []
        for m in fig_ref.finditer(block):
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) else lo
            mentioned.extend(range(lo, hi + 1))
        for n in sorted(set(mentioned)):
            if n in placed or n not in FIGURES:
                continue
            stem, caption = FIGURES[n]
            path = FIG_DIR / f"{stem}.png"
            if not path.exists():
                print(f"  ! missing figure file: {path} (skipping Figure {n})")
                continue
            rel = path.relative_to(PATHS["root"])
            out.append(f"![**Figure {n}.** {caption}]({rel})"
                       f"{{width={width_cm}cm}}")
            placed.add(n)

    missing = [n for n in FIGURES if n not in placed]
    if missing:
        out.append("\n## Figures not referenced in the text\n")
        for n in missing:
            stem, caption = FIGURES[n]
            path = FIG_DIR / f"{stem}.png"
            if path.exists():
                rel = path.relative_to(PATHS["root"])
                out.append(f"![**Figure {n}.** {caption}]({rel}){{width={width_cm}cm}}")
    return "\n\n".join(out)


def build(formats: List[str], width_cm: float, keep_md: bool = True) -> List[Path]:
    from datetime import date

    src = PATHS["paper"] / "paper_draft.md"
    if not src.exists():
        raise SystemExit("paper/paper_draft.md not found - run scripts/build_paper.py first.")
    if not FIG_DIR.exists():
        raise SystemExit(f"{FIG_DIR} not found - run "
                         "'generate_figures.py --manuscript' and "
                         "'explainability.py --manuscript' first.")

    md = src.read_text()
    # the H1 title becomes document metadata, so drop it from the body
    md = re.sub(r"^# .*?\n", "", md, count=1)
    body = YAML_HEADER.format(date=date.today().isoformat()) + insert_figures(md, width_cm)

    staged = PATHS["paper"] / "paper_with_figures.md"
    staged.write_text(body)

    produced = []
    for fmt in formats:
        out = PATHS["paper"] / f"RumenTwin-RL_manuscript.{fmt}"
        cmd = ["pandoc", str(staged), "-o", str(out),
               "--resource-path", str(PATHS["root"]),
               "--from", "markdown+pipe_tables+yaml_metadata_block+tex_math_dollars",
               "--standalone", "--toc", "--toc-depth=2"]
        if fmt == "pdf":
            engine = shutil.which("tectonic") or shutil.which("xelatex") or shutil.which("pdflatex")
            if not engine:
                print("  ! no LaTeX engine found - skipping PDF")
                continue
            cmd += [f"--pdf-engine={Path(engine).name}"]
        print(f"  pandoc -> {out.name}", flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-2000:]); print(res.stderr[-3000:])
            raise SystemExit(f"pandoc failed for {fmt}")
        if fmt == "docx":
            _set_a4(out)
        produced.append(out)
    if not keep_md:
        staged.unlink()
    return produced


def _set_a4(path: Path) -> None:
    """Pandoc's default reference document is US Letter; make it A4 so the Word
    and PDF versions have the same page geometry."""
    try:
        from docx import Document
        from docx.shared import Cm
    except ImportError:
        return
    doc = Document(str(path))
    for sec in doc.sections:
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.2)
        sec.top_margin = sec.bottom_margin = Cm(2.2)
    doc.save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DOCX/PDF manuscript with figures")
    ap.add_argument("--formats", nargs="*", default=["docx", "pdf"], choices=["docx", "pdf"])
    ap.add_argument("--width-cm", type=float, default=16.0)
    args = ap.parse_args()

    print(banner("RumenTwin-RL | building the conference manuscript"))
    print(DISCLAIMER)
    for p in build(args.formats, args.width_cm):
        size = p.stat().st_size / 1024
        print(f"  wrote {p.relative_to(PATHS['root'])}  ({size:,.0f} kB)")


if __name__ == "__main__":
    main()
