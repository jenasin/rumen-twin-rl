"""Shared publication figure style for RumenTwin-RL.

Design rules applied throughout (and worth stating, because they are checkable):

* **One axis per panel.** No dual-axis (twinx) plots anywhere; quantities with
  different units get their own stacked panel sharing the time axis.
* **Identity is never colour alone.** The four controllers carry a
  colour-vision-validated hue *and* a distinct marker shape *and* a distinct
  line style, plus a legend in every multi-series figure.
* **Colour by entity, not by rank.** The controller -> colour map is fixed in
  config.yaml, so a figure that drops a controller does not repaint the rest.
* **Recessive chrome.** Light y-grid beneath the data, no top/right spines,
  muted text; the data carries the ink.
* **A table view always exists.** Every figure has a machine-readable
  counterpart in ``tables/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .utils import DISCLAIMER, PATHS, load_config

CONTROLLER_LABEL = {
    "no_intervention": "No intervention",
    "expert_rules": "Expert rules",
    "dqn": "DQN",
    "ppo": "PPO",
}
# secondary (non-colour) encoding channels
CONTROLLER_MARKER = {"no_intervention": "o", "expert_rules": "s", "dqn": "^", "ppo": "D"}
CONTROLLER_LINESTYLE = {"no_intervention": (0, (1, 1.2)), "expert_rules": (0, (5, 1.6)),
                        "dqn": (0, (3, 1.2, 1, 1.2)), "ppo": "-"}
CONTROLLER_HATCH = {"no_intervention": "//", "expert_rules": "\\\\", "dqn": "..", "ppo": ""}

ACTION_MARKER = {1: "^", 2: "v", 3: "D", 4: "s", 5: "P", 6: "X"}

# Manuscript mode: figures destined for the paper drop their in-image titles and
# the standing disclaimer, because both are carried by the LaTeX/Word caption and
# the manuscript text. Standalone figures keep them so they are self-describing.
_MANUSCRIPT = {"on": False, "subdir": "manuscript"}


def set_manuscript_mode(on: bool = True, subdir: str = "manuscript") -> None:
    _MANUSCRIPT["on"] = bool(on)
    _MANUSCRIPT["subdir"] = subdir


def manuscript_mode() -> bool:
    return bool(_MANUSCRIPT["on"])


def get_style(config: Optional[Dict] = None) -> Dict:
    cfg = config or load_config()
    f = cfg["figures"]
    return {
        "palette": f["style_palette"],
        "status": f["status_palette"],
        "surface": f["surface"],
        "text_primary": f["text_primary"],
        "text_secondary": f["text_secondary"],
        "dpi": int(f["dpi"]),
        "formats": list(f["formats"]),
    }


def set_style(config: Optional[Dict] = None) -> Dict:
    """Apply the project-wide matplotlib rcParams. Returns the style dict."""
    st = get_style(config)
    plt.rcParams.update({
        "figure.facecolor": st["surface"],
        "axes.facecolor": st["surface"],
        "savefig.facecolor": st["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.0,
        "axes.labelcolor": st["text_primary"],
        "axes.edgecolor": "#c9c8c3",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e4e3de",
        "grid.linewidth": 0.7,
        "xtick.color": st["text_secondary"],
        "ytick.color": st["text_secondary"],
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8.0,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return st


def controller_colour(name: str, style: Dict) -> str:
    return style["palette"].get(name, "#52514e")


def controller_label(name: str) -> str:
    return CONTROLLER_LABEL.get(name, name)


def order_controllers(names: Iterable[str]) -> List[str]:
    order = ["no_intervention", "expert_rules", "dqn", "ppo"]
    names = list(names)
    return [c for c in order if c in names] + [c for c in names if c not in order]


def controller_legend(ax, names: Sequence[str], style: Dict, *, marker: bool = True,
                      loc: str = "best", ncol: int = 1, title: Optional[str] = None):
    """Legend carrying colour AND the secondary channel for each controller."""
    handles = [
        Line2D([0], [0], color=controller_colour(c, style),
               marker=CONTROLLER_MARKER.get(c, "o") if marker else None,
               linestyle=CONTROLLER_LINESTYLE.get(c, "-") if not marker else "none",
               markersize=5, linewidth=1.8, label=controller_label(c))
        for c in names
    ]
    return ax.legend(handles=handles, loc=loc, ncol=ncol, title=title)


def add_ph_threshold_bands(ax, cfg: Dict, style: Dict, *, label: bool = True,
                           xmin: float = 0.0, xmax: float = 1.0) -> None:
    """Shade the low- and severe-low-pH bands with the reserved status colours."""
    thr = cfg["thresholds"]
    lo, sev = float(thr["low_pH"]), float(thr["severe_low_pH"])
    ax.axhspan(sev, lo, color=style["status"]["warning"], alpha=0.13, zorder=0, lw=0)
    ax.axhspan(ax.get_ylim()[0], sev, color=style["status"]["critical"], alpha=0.13, zorder=0, lw=0)
    ax.axhline(lo, color=style["status"]["warning"], lw=1.0, ls="--", zorder=1)
    ax.axhline(sev, color=style["status"]["critical"], lw=1.0, ls="--", zorder=1)
    if label:
        ax.annotate(f"low pH {lo:g}", xy=(0.006, lo), xycoords=("axes fraction", "data"),
                    xytext=(0, 2), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7, color="#8a6400")
        ax.annotate(f"severe {sev:g}", xy=(0.006, sev), xycoords=("axes fraction", "data"),
                    xytext=(0, 2), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7, color="#9c2b2b")


def bar_with_ci(ax, x: np.ndarray, means: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                colours: Sequence[str], *, width: float = 0.7,
                hatches: Optional[Sequence[str]] = None, label_fmt: Optional[str] = None,
                text_colour: str = "#0b0b0b") -> None:
    """Bars anchored at the baseline with asymmetric 95 % CI whiskers."""
    err = np.vstack([np.clip(means - lo, 0, None), np.clip(hi - means, 0, None)])
    bars = ax.bar(x, means, width=width, color=colours, edgecolor=plt.rcParams["figure.facecolor"],
                  linewidth=1.2, zorder=2)
    if hatches is not None:
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
    ax.errorbar(x, means, yerr=err, fmt="none", ecolor="#3a3a38", elinewidth=1.1,
                capsize=2.5, capthick=1.1, zorder=3)
    if label_fmt:
        for xi, m, h in zip(x, means, hi):
            ax.annotate(label_fmt.format(m), xy=(xi, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=text_colour)


def footer(fig, extra: str = "") -> None:
    txt = DISCLAIMER + ((" " + extra) if extra else "")
    fig.text(0.5, -0.012, txt, ha="center", va="top", fontsize=6.6, color="#6d6c68")


def save_figure(fig, name: str, config: Optional[Dict] = None,
                disclaimer: bool = True, extra_footer: str = "") -> List[Path]:
    """Write a figure to figures/ in every configured format."""
    cfg = config or load_config()
    st = get_style(cfg)
    out_dir = PATHS["figures"]
    if _MANUSCRIPT["on"]:
        # Strip only the "Figure N | ..." headline (it is carried by the
        # document caption); per-panel subtitles are content and must stay.
        # savefig(bbox_inches="tight") then crops the whitespace left behind.
        for ax in fig.axes:
            # titles set with loc="left"/"right" are NOT returned by the
            # default get_title(), so every location has to be checked
            for loc in ("center", "left", "right"):
                if ax.get_title(loc=loc).startswith("Figure "):
                    ax.set_title("", loc=loc)
        sup = getattr(fig, "_suptitle", None)
        if sup is not None and sup.get_text().startswith("Figure "):
            sup.set_text("")
        disclaimer = False
        out_dir = out_dir / _MANUSCRIPT["subdir"]
    if disclaimer:
        footer(fig, extra_footer)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in st["formats"]:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=st["dpi"], format=ext)
        out.append(p)
    plt.close(fig)
    print(f"  wrote {', '.join(p.name for p in out)}")
    return out
