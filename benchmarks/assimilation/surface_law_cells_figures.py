"""Figures of the per-cell runs of group 12 of the Savannah twin (log-law surface coefficient).

One Manning's n per cell (zones not given), with the surface velocity tied to n and depth by a
log-law profile in the data and in the fit: no noise, noise with regularization weight 1, 10 and
100, and noise with a prior on n. Two figures in ``<runs_dir>/figures``:

- ``q12_cells_overview.png``: maps of the fitted n, the error of the discharge of every run
  (with the zone fits, the constant coefficient and the index velocity for comparison), and the
  error of the discharge during the fit;
- ``q12_cells_strip.png``: across the river under the camera strip, the fitted n, the surface
  coefficient it implies, and the depth-averaged speed, against the truth.

Usage::

    python benchmarks/assimilation/surface_law_cells_figures.py [<runs_dir>]

``<runs_dir>`` defaults to ``cases/savannah/assimilation``. Needs ``savannah_figures.py`` and
``savannah_summary.py`` (same folder, or on ``PYTHONPATH``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iteration_history  # noqa: E402
from savannah_figures import (  # noqa: E402
    AQUA,
    ORANGE,
    YELLOW,
    Twin,
    clean,
    colourbar,
    save,
    style,
)
from savannah_summary import (  # noqa: E402
    BLUE,
    GRID,
    INK,
    INK2,
    MUTED,
    SEQUENTIAL,
    SURFACE,
    draw_names,
    find_run,
    q_error_pct,
)

from hydrograd.apps.assimilation import log_law_surface_factor, velocities  # noqa: E402

MAGENTA = "#e87ba4"                                         # categorical slot 5
BOUNDS = (0.015, 0.15)
SITE = (473.0, 156.0)                                       # the mid camera site
CASES = [                                                   # label, run name, colour
    ("no noise, weight 1", "cam_mid_1g_cells_ll_fitll", MAGENTA),
    ("noise, weight 1", "cam_mid_1g_cells_r1_ll_fitll_noise_{d}", BLUE),
    ("noise, weight 10", "cam_mid_1g_cells_r10_ll_fitll_noise_{d}", ORANGE),
    ("noise, weight 100", "cam_mid_1g_cells_r100_ll_fitll_noise_{d}", AQUA),
    ("noise, weight 1, prior on n", "cam_mid_1g_cells_prior0.3_ll_fitll_noise_{d}", YELLOW),
]
REFERENCES = [                                              # label, run name, summary key, marker
    ("one n per zone, log law (100 iterations)", "cam_mid_1g_ll_fitll_noise_{d}", "Q", "o"),
    ("one n per zone, log law, prior on n", "cam_mid_1g_ll_fitll_prior0.3_noise_{d}", "Q", "o"),
    ("one n per cell, constant coefficient\n(group 11, weight 1)", "cam_mid_1g_cells_noise_{d}",
     "Q", "s"),
    ("index velocity, coefficient 0.85", "cam_mid_1g_cells_r1_ll_fitll_noise_{d}",
     "q_index_m3s", "D"),
    ("first guess of the fit (n = 0.04)", "cam_mid_1g_cells_r1_ll_fitll_noise_{d}",
     "q_first_guess_m3s", "x"),
]


def folders(twin: Twin, name: str) -> list[Path]:
    """Folders of a run, or of its noise draws."""
    return [f for n in draw_names(name) if (f := find_run(twin.runs, n)) is not None]


def summary(folder: Path) -> dict:
    return json.loads((folder / "summary.json").read_text())


def errors_text(values: list[float]) -> str:
    return ", ".join(f"{v:+.1f}" for v in values) + " %"


# ----------------------------------------------------------------------
# Overview: maps, discharge error of every run, discharge error during the fit
# ----------------------------------------------------------------------
def overview(twin: Twin, plt, slurm_logs: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    fig = plt.figure(figsize=(17.0, 8.6))
    grid = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 0.045, 0.22, 1.5], wspace=0.12,
                            hspace=0.2, left=0.02, right=0.98, top=0.86, bottom=0.075)

    # Maps of n around the strip: the truth and draw 1 of each case.
    sequential = LinearSegmentedColormap.from_list("manning", SEQUENTIAL)
    scale = LogNorm(*BOUNDS)
    window = (SITE[0], SITE[1], 210.0)
    panels = [("true Manning's n", None)] + [(label, name) for label, name, _ in CASES]
    for k, (label, name) in enumerate(panels):
        ax = fig.add_subplot(grid[k // 3, k % 3])
        if name is None:
            twin.map(ax, twin.n_true, sequential, scale, label, window)
        elif runs := folders(twin, name):
            n_fit = np.load(runs[0] / "reconstruction.npz")["manning_cells"]
            rms = 100 * np.sqrt(np.mean((n_fit / twin.n_true - 1) ** 2))
            draw = ", draw 1" if len(runs) > 1 else ""
            twin.map(ax, n_fit, sequential, scale,
                     f"{label}{draw}\nn {n_fit.min():.3f} to {n_fit.max():.3f}, "
                     f"{rms:.0f} % RMS off, Q {q_error_pct(summary(runs[0])):+.1f} %", window)
            twin.gauges(ax, runs[0])
        ax.set_xticks([])
        ax.set_yticks([])
    bar = fig.add_subplot(grid[:, 3])
    colourbar(fig, bar, sequential, scale, "Manning's n (s/m$^{1/3}$)",
              ticks=[0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15], format="%.3g")
    bar.minorticks_off()

    # Error of the discharge: every run of the group, then what to compare it with.
    ax = fig.add_subplot(grid[0, 5])
    box = ax.get_position()
    ax.set_position([box.x0 + 0.42 * box.width, box.y0, 0.58 * box.width, box.height])
    y, ticks = 0, []

    def header(text: str) -> None:
        nonlocal y
        ax.text(-0.72, y + 0.1, text, transform=ax.get_yaxis_transform(), fontsize=9.5,
                fontweight="bold", va="center", ha="left", clip_on=False)
        y += 1

    header("One n per cell, log law (this group)")
    for label, name, colour in CASES:
        values = [q_error_pct(summary(f)) for f in folders(twin, name)]
        ax.plot(values, [y] * len(values), "s", color=colour, ms=7.5, mec=SURFACE, mew=0.8, ls="",
                zorder=3)
        ticks.append((y, label))
        y += 1
    header("For comparison, same noise (3 draws)")
    for label, name, key, marker in REFERENCES:
        values = [q_error_pct(summary(f), key) for f in folders(twin, name)]
        ax.plot(values, [y] * len(values), marker, color=MUTED, ms=6.5, mew=1.6 if marker == "x"
                else 0.8, mec=MUTED if marker == "x" else SURFACE, ls="", zorder=3)
        ticks.append((y, label))
        y += 1
    ax.set_ylim(y - 0.4, -0.7)
    ax.set_yticks([t for t, _ in ticks], [label for _, label in ticks], color=INK2, fontsize=8.5)
    ax.axvline(0.0, color="#c3c2b7", lw=1.0)
    ax.set_xlabel("error of the discharge at the end of the fit (%)")
    clean(ax, "x")

    # Error of the discharge during the fit, from the logs.
    ax = fig.add_subplot(grid[1, 5])
    box = ax.get_position()
    ax.set_position([box.x0 + 0.1 * box.width, box.y0 + 0.2 * box.height, 0.9 * box.width,
                     0.72 * box.height])
    pool = [b for f in sorted(slurm_logs.glob("*.out"), reverse=True)
            for b in iteration_history.parse_blocks(f)]
    for label, name, colour in CASES:
        for k, folder in enumerate(folders(twin, name)):
            run = iteration_history.load_run(folder, pool)
            if run["log"] is None:
                continue
            q_true = float(np.mean(run["summary"]["Q_true"]))
            ax.plot(run["log"]["iteration"], 100 * (run["log"]["q_max"] / q_true - 1),
                    color=colour, lw=1.6, label=label if k == 0 else None)
    ax.axhline(0.0, color="#c3c2b7", lw=1.0)
    ax.set_ylim(-21, 9)
    ax.set_xlim(0, 1000)
    ax.set_xlabel("iteration")
    ax.set_ylabel("error of the discharge (%)")
    ax.set_title("During the fit (the runs with the prior stop by themselves)", fontsize=9.5,
                 loc="left", color=INK)
    clean(ax, "y")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(-0.02, -0.17), ncols=2,
              handlelength=1.4, columnspacing=1.2)

    fig.suptitle("One Manning's n per cell with the surface coefficient from a log-law profile "
                 "(data and fit): only the prior on n keeps the discharge\nCamera upstream of the "
                 "bend, 46 cells with surface velocity (black dots), stage at the camera "
                 "(triangle); noise 5 cm/s and 10 mm; n bounded to 0.015-0.15, first guess 0.04; "
                 "prior: SD of log n 0.3 around 0.04", fontsize=11, x=0.02, ha="left")
    save(fig, twin.runs, "q12_cells_overview.png")


# ----------------------------------------------------------------------
# Under the strip: n, the coefficient it implies, and the depth-averaged speed
# ----------------------------------------------------------------------
def strip(twin: Twin, plt) -> None:
    first = folders(twin, CASES[0][1])
    if not first:
        return
    cells = np.load(first[0] / "observations.npz")["cells"]
    xy = twin.centroid[cells]
    # Distance across the river: along the long axis of the strip, from the camera bank.
    axis = np.linalg.svd(xy - xy.mean(axis=0))[2][0]
    camera_xy = np.asarray(summary(first[0])["camera"][0])
    s = (xy - camera_xy) @ axis
    if s.mean() < 0:
        s = -s
    edges = np.arange(np.floor(s.min() / 10) * 10, s.max() + 10, 10.0)
    centre = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(s, edges) - 1

    def binned(values: np.ndarray) -> np.ndarray:
        return np.array([values[which == k].mean() if np.any(which == k) else np.nan
                         for k in range(len(centre))])

    def fields(n: np.ndarray, h: np.ndarray, u: np.ndarray, v: np.ndarray) -> list[np.ndarray]:
        """n, mean / surface velocity of the log law, depth-averaged speed at the strip cells."""
        ratio = np.asarray(log_law_surface_factor(n[cells], h[cells], twin.m.g))
        return [n[cells], 1.0 / ratio, np.hypot(u, v)[cells]]

    truth = fields(twin.n_true, twin.h, twin.u, twin.v)
    rows = [("Manning's n (s/m$^{1/3}$)", "fitted n"),
            ("depth-averaged / surface velocity", "surface coefficient of the log law"),
            ("depth-averaged speed (m/s)", "depth-averaged speed")]

    fig, axes = plt.subplots(3, len(CASES), figsize=(17.0, 8.8), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.83, bottom=0.075, wspace=0.07, hspace=0.16)
    fig.supxlabel("distance across the river from the camera bank (m)", fontsize=9.5, color=INK2,
                  y=0.015)
    for j, (label, name, colour) in enumerate(CASES):
        runs = folders(twin, name)
        q_err = [q_error_pct(summary(f)) for f in runs]
        for i, (ylabel, _) in enumerate(rows):
            ax = axes[i, j]
            for folder in runs:
                rec = np.load(folder / "reconstruction.npz")
                h, u, v = (np.asarray(a) for a in velocities(twin.m, rec["ys"][-1]))
                fit = fields(rec["manning_cells"], h, u, v)[i]
                ax.plot(s, fit, ".", color=colour, ms=4, alpha=0.3, zorder=2)
                ax.plot(centre, binned(fit), "-", color=colour, lw=1.8, zorder=3)
            ax.plot(centre, binned(truth[i]), "--", color=INK, lw=1.6, zorder=4)
            if i == 0:
                ax.set_yscale("log")
                for bound in BOUNDS:
                    ax.axhline(bound, color=MUTED, lw=0.9, ls=":")
                ax.set_yticks([0.015, 0.02, 0.03, 0.05, 0.1, 0.15],
                              ["0.015", "0.02", "0.03", "0.05", "0.1", "0.15"])
                ax.minorticks_off()
                ax.set_title(f"{label}\nQ {errors_text(q_err)}", fontsize=10, loc="left",
                             color=INK)
            if j == 0:
                ax.set_ylabel(ylabel)
            clean(ax, "y")
    axes[0, 0].text(centre[-1], BOUNDS[1] * 0.93, "bounds of n", fontsize=8, color=MUTED,
                    ha="right", va="top")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK, lw=1.6, ls="--", label="truth (mean of 10 m bins)"),
               Line2D([], [], color=INK2, lw=1.8, label="fit, one line per noise draw "
                                                         "(mean of 10 m bins)"),
               Line2D([], [], color=INK2, marker=".", ms=6, alpha=0.5, ls="",
                      label="fit, the 46 cells of the strip")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), fontsize=9,
               ncols=3)
    fig.suptitle("Under the camera strip: the fit raises n where it has data, which lowers the "
                 "surface coefficient and the depth-averaged speed, and with it the discharge\n"
                 "One Manning's n per cell, surface coefficient from the log law of the fitted n "
                 "and depth; at weights 1 and 10 cells of the strip reach the upper bound of n", fontsize=11,
                 x=0.055, y=0.94, ha="left")
    save(fig, twin.runs, "q12_cells_strip.png")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cases/savannah/assimilation")
    style(plt)
    twin = Twin(runs)
    if not folders(twin, CASES[0][1]):
        sys.exit(f"no per-cell run of group 12 under {runs}")
    overview(twin, plt, runs.parent / "logs")
    strip(twin, plt)


if __name__ == "__main__":
    main()
