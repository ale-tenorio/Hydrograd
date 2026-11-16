"""Per-iteration comparison plots of the assimilation runs, from the logs already on disk.

No run is repeated. Sources, per run directory:

- the text log (``run.log`` in the run directory, ``<group>/logs/<run>.log``, or a block of a
  Slurm ``da-*.out`` file): one line per iteration with the total loss, the velocity and gauge
  terms, the discharge (min-max over the knots), Manning's n where fitted, and the elapsed time;
- ``summary.json``: loss history, first-guess and final discharge, final exit stage, errors
  against the truth;
- ``observations.npz``: the gauge sigma and the data counts, to turn the gauge term of the loss
  into an RMS stage residual at the gauge(s).

The exit stage was not written per iteration (``progress.json`` keeps only the last record), so
the stage panel shows the RMS residual at the gauge(s) per iteration, which the logs do give,
and the final exit-stage error appears in the fit figure.

A group that mixes Manning's n per zone and per cell is split into ``<group>_zones`` and
``<group>_cells``. Per run group, two figures: ``<group>_iterations.png`` and ``<group>_fit.png``; per case, one
``loss_vs_error.png`` over all its runs.

Usage:
    python benchmarks/assimilation/iteration_history.py [--cases cases]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ITER = re.compile(r"^iter\s+(\d+): loss (\S+) \(velocity (\S+), gauge (\S+)\), "
                  r"Q ([\d.]+)-([\d.]+) m\^3/s(.*), (\d+) s\s*$")
DRAW = re.compile(r"(_noise)?_d\d+$")
N_FLOOR = 0.1      # %, floor of the zone-n error plot (the logs give n to 4 decimals)


def positive(x: np.ndarray) -> np.ndarray:
    """x with its non-positive entries masked, for a log axis (a zero term: no such data)."""
    return np.where(x > 0, x, np.nan)


def parse_blocks(path: Path) -> list[dict[str, np.ndarray]]:
    """Iteration blocks of a log file (a new block starts at every ``iter 0``)."""
    blocks: list[list[tuple]] = []
    for line in path.read_text(errors="replace").splitlines():
        m = ITER.match(line)
        if not m:
            continue
        k = int(m.group(1))
        if k == 0 or not blocks:
            blocks.append([])
        n_values: list[float] = []
        n_range = (np.nan, np.nan)
        for piece in m.group(7).split(", "):
            if piece.startswith("n ") and "-" in piece:
                lo, hi = piece[2:].split("-")
                n_range = (float(lo), float(hi))
            elif piece.startswith("n "):
                n_values = [float(x) for x in piece[2:].split()]
        blocks[-1].append((k, float(m.group(2)), float(m.group(3)), float(m.group(4)),
                           float(m.group(5)), float(m.group(6)), n_values, n_range,
                           float(m.group(8))))
    out = []
    for b in blocks:
        out.append({
            "iteration": np.array([r[0] for r in b]), "loss": np.array([r[1] for r in b]),
            "velocity": np.array([r[2] for r in b]), "gauge": np.array([r[3] for r in b]),
            "q_min": np.array([r[4] for r in b]), "q_max": np.array([r[5] for r in b]),
            "n": np.array([r[6] for r in b]) if b[0][6] else None,
            "n_range": np.array([r[7] for r in b]) if np.isfinite(b[0][7][0]) else None,
            "elapsed": np.array([r[8] for r in b]),
        })
    return out


def load_run(run: Path, pool: list[dict]) -> dict:
    """Summary, log block (or None) and gauge scaling of one run directory."""
    s = json.loads((run / "summary.json").read_text())
    hist = np.asarray(s["loss_history"], dtype=float)
    candidates: list[dict] = []
    for log in (run / "run.log", run.parent / "logs" / f"{run.name}.log"):
        if log.exists():
            candidates += parse_blocks(log)
    candidates += pool

    def misfit(b: dict) -> float:
        """Largest relative difference between a log block and the loss history of this run,
        over the iterations they share. Runs on the same data with other settings (weight,
        prior) start from the same loss, so the first value alone does not tell them apart."""
        m = min(len(b["loss"]), len(hist))
        return float(np.max(np.abs(b["loss"][:m] - hist[:m]) / np.abs(hist[:m])))

    # The log prints the loss to 5 digits. Blocks of the full length first, then the closest.
    close = [b for b in candidates if misfit(b) <= 2e-4]
    block = min(close, key=lambda b: (len(b["loss"]) != len(hist), misfit(b)), default=None)

    # Gauge term of the loss -> RMS stage residual at the gauge(s), mm.
    obs = np.load(run / "observations.npz")
    n_gauge, n_samples = obs["gauge_wse"].size, obs["u"].size
    scale = 1.0 if s.get("weighting", "type") == "type" else (2 * n_samples + n_gauge) / n_gauge
    return {"name": run.name, "summary": s, "log": block, "loss_history": hist,
            "gauge_to_mm": lambda g: 1000.0 * float(obs["sigma_wse"]) * np.sqrt(g * scale)}


def styles(runs: list[dict]) -> dict[str, dict]:
    """One colour per run, noise draws of the same setup sharing it."""
    bases = sorted({DRAW.sub("", r["name"]) for r in runs})
    cmap = plt.get_cmap("tab20" if len(bases) > 10 else "tab10")
    dashes = ["-", "--", ":", "-."]
    look = {b: (cmap(i % cmap.N), dashes[(i // cmap.N) % 4]) for i, b in enumerate(bases)}
    seen: set[str] = set()
    out = {}
    for r in runs:
        base = DRAW.sub("", r["name"])
        n_draws = sum(DRAW.sub("", q["name"]) == base for q in runs)
        label = None if base in seen else (base + (f" ({n_draws} draws)" if n_draws > 1 else ""))
        seen.add(base)
        out[r["name"]] = {"color": look[base][0], "ls": look[base][1], "label": label,
                          "lw": 1.0 if n_draws > 1 else 1.5, "alpha": 0.8}
    return out


def manning_series(r: dict) -> tuple[str, np.ndarray] | None:
    """Per-iteration Manning's n of a run: ('single' | 'zones' | 'cells', series)."""
    log, s = r["log"], r["summary"]
    if log is None:
        return None
    if log["n_range"] is not None:
        return "cells", log["n_range"]
    if log["n"] is None or not s.get("fit_manning"):
        return None
    true = np.asarray(s["manning_true"], dtype=float)
    if log["n"].shape[1] == 1:
        return "single", 100.0 * (log["n"][:, 0] / true[0] - 1.0)
    true = true[np.asarray(s["zones"])]
    return "zones", 100.0 * np.sqrt(np.mean((log["n"] / true - 1.0) ** 2, axis=1))


def plot_iterations(title: str, runs: list[dict], out: Path) -> None:
    """Loss, loss terms, discharge, gauge residual and Manning's n against the iteration."""
    look = styles(runs)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    ax_loss, ax_vel, ax_gauge, ax_q, ax_n, ax_time = axes.ravel()
    steady = all(r["summary"]["steady"] for r in runs)
    no_log = [r["name"] for r in runs if r["log"] is None]
    modes = set()

    for r in runs:
        s, log, kw = r["summary"], r["log"], dict(look[r["name"]])
        q_true = np.asarray(s["Q_true"], dtype=float)
        if log is None:      # loss history and the two end points of Q only
            ax_loss.semilogy(r["loss_history"], **kw)
            if steady:
                kw.update(label=None, marker="o", ms=3)
                ax_q.plot([0, s["iterations"]],
                          100 * (np.array([s["q_first_guess_m3s"], s["Q"][0]]) / q_true[0] - 1),
                          **kw)
            continue
        it = log["iteration"]
        ax_loss.semilogy(it, log["loss"], **kw)
        kw["label"] = None
        ax_vel.semilogy(it, positive(log["velocity"]), **kw)
        ax_gauge.semilogy(it, positive(r["gauge_to_mm"](log["gauge"])), **kw)
        ax_time.semilogy(log["elapsed"], log["loss"], **kw)
        if steady:
            ax_q.plot(it, 100 * (log["q_max"] / q_true[0] - 1), **kw)
        else:
            ax_q.plot(it, log["q_max"], **kw)
            ax_q.plot(it, log["q_min"], **{**kw, "lw": 0.8, "alpha": 0.5})
        series = manning_series(r)
        if series is not None:
            modes.add(series[0])
            if series[0] == "cells":
                ax_n.semilogy(it, series[1][:, 1], **kw)
                ax_n.semilogy(it, series[1][:, 0], **{**kw, "lw": 0.8, "alpha": 0.5})
            elif series[0] == "zones":
                ax_n.semilogy(it, np.maximum(series[1], N_FLOOR), **kw)
            else:
                ax_n.plot(it, series[1], **kw)

    ax_loss.set(title="Total loss", ylabel="J")
    ax_vel.set(title="Velocity term of the loss", ylabel="J velocity")
    ax_gauge.set(title="Stage: RMS residual at the gauge(s)\n(exit stage was not logged per "
                       "iteration)", ylabel="mm")
    ax_time.set(title="Total loss against wall time", xlabel="elapsed (s)", ylabel="J")
    if steady:
        ax_q.axhline(0, color="k", lw=0.8)
        ax_q.set_yscale("symlog", linthresh=1.0)
        ax_q.set(title="Discharge error against the truth (linear within 1 %)", ylabel="%")
    else:
        q_true = np.asarray(runs[0]["summary"]["Q_true"], dtype=float)
        for q in (q_true.min(), q_true.max()):
            ax_q.axhline(q, color="k", lw=0.8, ls=":")
        ax_q.set(title="Discharge: largest (bold) and smallest (thin) knot value\n"
                       "dotted: true peak and base flow", ylabel="m$^3$/s")
    if modes == {"cells"}:
        true = np.asarray(runs[0]["summary"]["manning_true"], dtype=float)
        zones = runs[0]["summary"].get("zones")
        true = true[np.asarray(zones)] if zones and len(true) > max(zones) else true
        for n in (true.min(), true.max()):
            ax_n.axhline(n, color="k", lw=0.8, ls=":")
        ax_n.set(title="Manning's n per cell: largest (bold) and smallest (thin)\n"
                       "dotted: true range", ylabel="n")
    elif modes == {"zones"}:
        ax_n.set(title="Manning's n: RMS error over the zones\n"
                       f"(floor {N_FLOOR} %: n is logged to 4 decimals)", ylabel="%")
    elif modes == {"single"}:
        ax_n.axhline(0, color="k", lw=0.8)
        ax_n.set(title="Manning's n: error against the truth", ylabel="%")
    elif modes:
        ax_n.set(title="Manning's n: " + " / ".join(sorted(modes)) + " (mixed units)")
    else:
        ax_n.text(0.5, 0.5, "Manning's n not fitted\n(or not in the logs)", ha="center",
                  va="center", transform=ax_n.transAxes, color="0.4")
        ax_n.set(title="Manning's n")
    for a in (ax_loss, ax_vel, ax_gauge, ax_q, ax_n):
        a.set_xlabel("iteration")
    for a in axes.ravel():
        a.grid(alpha=0.3, which="both", lw=0.4)

    note = f"\nno iteration log (loss history and end points only): {', '.join(no_log)}" \
        if no_log else ""
    fig.suptitle(f"{title}: {len(runs)} runs{note}", fontsize=13)
    handles, labels = ax_loss.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(6, len(labels)),
               fontsize=8, frameon=False)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def manning_error(s: dict) -> float:
    """Final error of Manning's n (%): RMS over cells or zones, signed for a single value."""
    if not s.get("fit_manning"):
        return np.nan
    if "manning_cell_rmse_pct" in s:
        return s["manning_cell_rmse_pct"]
    return 100.0 * (s["manning"][0] / s["manning_true"][0] - 1.0)


def plot_fit(title: str, runs: list[dict], out: Path) -> None:
    """Final fit of every run against the truth, with the first guess for reference."""
    look = styles(runs)
    names = [r["name"] for r in runs]
    y = np.arange(len(runs))[::-1]
    colors = [look[n]["color"] for n in names]
    S = [r["summary"] for r in runs]
    steady = all(s["steady"] for s in S)
    fit_n = any(s.get("fit_manning") for s in S)
    fig, axes = plt.subplots(1, 5 if fit_n else 4, sharey=True, constrained_layout=True,
                             figsize=(17, max(3.5, 0.2 * len(runs) + 1.8)))

    q_true = [float(np.mean(s["Q_true"])) for s in S]
    if steady:
        q = [100 * (s["Q"][0] / t - 1) for s, t in zip(S, q_true)]
        q0 = [100 * (s["q_first_guess_m3s"] / t - 1) for s, t in zip(S, q_true)]
        axes[0].barh(y, q, color=colors)
        axes[0].plot(q0, y, "k|", ms=8, label="first guess")
        axes[0].set_xscale("symlog", linthresh=1.0)
        axes[0].set(title="Discharge error (%)")
        w = [1000 * (s["WSE"][0] - s["WSE_true"]) for s in S]
        axes[1].barh(y, w, color=colors)
        axes[1].set_xscale("symlog", linthresh=1.0)
        axes[1].set(title="Exit stage error (mm)")
    else:
        axes[0].barh(y, [100 * s["q_rmse_m3s"] / t for s, t in zip(S, q_true)], color=colors)
        axes[0].set_xscale("log")
        axes[0].set(title="Discharge RMSE over the window\n(% of the mean true Q)")
        axes[1].barh(y, [s["wse_exit_rmse_mm"] for s in S], color=colors)
        axes[1].set_xscale("log")
        axes[1].set(title="Exit stage RMSE (mm)")
    axes[2].barh(y, [100 * s["errors"]["rmse_U"] for s in S], color=colors)
    axes[2].plot([100 * s["errors_first_guess"]["rmse_U"] for s in S], y, "k|", ms=8,
                 label="first guess")
    axes[2].set_xscale("log")
    axes[2].set(title="Velocity RMSE, whole field (cm/s)")
    axes[3].barh(y, [s["loss_final"] for s in S], color=colors)
    axes[3].plot([s["loss_first"] for s in S], y, "k|", ms=8, label="first guess")
    axes[3].set_xscale("log")
    axes[3].set(title="Loss J")
    if fit_n:
        axes[4].barh(y, [manning_error(s) for s in S], color=colors)
        axes[4].set_xscale("symlog", linthresh=1.0)
        axes[4].set(title="Manning's n error (%)\nRMS over cells; signed if one value")
    for a in axes:
        a.axvline(0, color="k", lw=0.6)
        a.set_axisbelow(True)
        a.grid(alpha=0.3, axis="x", which="both", lw=0.4)
    axes[0].set_yticks(y, names, fontsize=7)
    axes[0].set_ylim(-0.7, len(runs) - 0.3)
    axes[0 if steady else 2].legend(fontsize=7, loc="best")
    fig.suptitle(f"{title}: final fit against the truth", fontsize=13)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_loss_vs_error(case: str, groups: dict[str, list[dict]], out: Path) -> None:
    """Does a low loss mean a good field? Final loss against the whole-field velocity error."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab20")
    for i, (group, runs) in enumerate(groups.items()):
        S = [r["summary"] for r in runs]
        noisy = np.array([s["noise_u_cm_s"] > 0 or s["noise_wse_mm"] > 0 for s in S])
        loss = np.array([s["loss_final"] for s in S])
        for ax, key in zip(axes, ("rmse_U", "rmse_U_observed")):
            err = np.array([100 * s["errors"][key] for s in S])
            ax.loglog(loss[~noisy], err[~noisy], "o", color=cmap(i % 20), ms=5, label=group)
            ax.loglog(loss[noisy], err[noisy], "x", color=cmap(i % 20), ms=5)
    axes[0].set(title="Whole field", xlabel="final loss J", ylabel="velocity RMSE (cm/s)")
    axes[1].set(title="Observed cells only", xlabel="final loss J",
                ylabel="velocity RMSE (cm/s)")
    for a in axes:
        a.grid(alpha=0.3, which="both", lw=0.4)
    fig.suptitle(f"{case}: final loss against the velocity error of every run "
                 "(circle: exact data, cross: noisy data)", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", type=Path, default=Path("cases"))
    parser.add_argument("--only", default="", help="only the run groups whose name has this")
    args = parser.parse_args()

    layout = {      # case -> (directory of the run groups, shared Slurm logs, output directory)
        "block_in_channel": (args.cases / "block_in_channel",
                             args.cases / "block_in_channel" / "logs",
                             args.cases / "block_in_channel" / "iteration_plots"),
        "savannah": (args.cases / "savannah" / "assimilation" / "runs",
                     args.cases / "savannah" / "logs",
                     args.cases / "savannah" / "assimilation" / "iteration_plots"),
    }
    for case, (root, slurm, out) in layout.items():
        pool = [b for f in sorted(slurm.glob("*.out"), reverse=True) for b in parse_blocks(f)]
        groups: dict[str, list[dict]] = {}
        for group in sorted(root.iterdir(), key=lambda p: (len(p.name), p.name)):
            dirs = sorted(d for d in group.glob("*") if (d / "observations.npz").exists()
                          and (d / "summary.json").exists()) if group.is_dir() else []
            if args.only and args.only not in group.name:
                continue
            runs = [load_run(d, pool) for d in dirs]
            # A group that mixes Manning's n per zone and per cell is split in two.
            modes = sorted({r["summary"].get("fit_manning_mode") or "fixed" for r in runs})
            if {"zones", "cells"} <= set(modes):
                for mode in modes:
                    groups[f"{group.name}_{mode}"] = [
                        r for r in runs
                        if (r["summary"].get("fit_manning_mode") or "fixed") == mode]
            elif runs:
                groups[group.name] = runs
        out.mkdir(parents=True, exist_ok=True)
        for group, runs in groups.items():
            short = [r for r in runs
                     if r["log"] is not None and len(r["log"]["loss"]) != len(r["loss_history"])]
            print(f"{case}/{group}: {len(runs)} runs, "
                  f"{sum(r['log'] is not None for r in runs)} with an iteration log"
                  + (f", log length differs from the loss history: "
                     f"{[r['name'] for r in short]}" if short else ""))
            plot_iterations(f"{case} / {group}", runs, out / f"{group}_iterations.png")
            plot_fit(f"{case} / {group}", runs, out / f"{group}_fit.png")
        if not args.only:
            plot_loss_vs_error(case, groups, out / "loss_vs_error.png")
        print(f"figures in {out}")


if __name__ == "__main__":
    main()
