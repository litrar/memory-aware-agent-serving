"""Plots for one experiment group (results/<group>-<TS>/), where each experiment ran at a single fixed C.

experiment.py's curves.png puts concurrency on the x-axis, so it cannot compare fixed-C experiments.
This script compares the experiments of a group side by side and draws a timeline of every run:

  curves_group.png             one bar/dot per experiment: tasks, throughput, completion and queueing time,
                               server memory, context tokens vs KV capacity, LLM request time split,
                               prompt-cache reuse (values from summary_all.csv)
  timelines/<experiment>.png   per run over time: each task's queue / LLM / tool segments, slot context
                               tokens against n_ctx, and server memory (from events.jsonl / memory.jsonl)

Usage (from the repo root):  python -B scripts/plot_group.py results/H-conditions-<TS> [more groups...]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import GRID, INK, INK_2, SERIES, SURFACE      # same look as curves.png

QUEUE = "#d9d8d4"           # recessive neutral for time spent waiting in our FIFO queue


def natural(s):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]


def num(row, key, scale=1.0):
    v = row.get(key)
    try:
        return None if v in (None, "") else float(v) * scale
    except ValueError:
        return None


def style():
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                         "ytick.color": INK_2, "axes.edgecolor": GRID, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})


def tidy(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def legend_above(ax):
    if not ax.get_legend_handles_labels()[0]:       # nothing plotted (e.g. no memory samples in a test run)
        return
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8,
              borderaxespad=0.2, handlelength=1.5).set_in_layout(False)


def label_bars(ax, xs, ys, fmt):
    """Value above each bar in text ink (identity stays with the bar, values with the text)."""
    top = max([y for y in ys if y is not None] or [0])
    for x, y in zip(xs, ys):
        if y is not None:
            ax.annotate(fmt(y), (x, y), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8, color=INK_2)
    ax.set_ylim(0, top * 1.18 if top else 1)


# ---------------------------------------------------------------------------- group comparison

def plot_group(rows, path, title):
    names = [r["experiment"] for r in rows]
    xs = list(range(len(rows)))
    style()
    fig, axes = plt.subplots(2, 4, figsize=(max(18, 1.5 * len(rows)), 8.6))
    rotate = 30 if len(rows) > 4 or max(map(len, names)) > 12 else 0
    for ax in axes.flat:
        tidy(ax)
        ax.set_xticks(xs, names, rotation=rotate, ha="right" if rotate else "center")
        ax.set_xlim(-0.6, len(rows) - 0.4)

    def bars(ax, title, key, fmt, scale=1.0):
        ys = [num(r, key, scale) for r in rows]
        ax.bar(xs, [y or 0 for y in ys], width=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        label_bars(ax, xs, ys, fmt)
        ax.set_title(title, pad=22)

    def paired(ax, title, series, fmt, ref=None):
        """Two or three measures in the same unit as grouped bars, a legend above, values on top."""
        k = len(series)
        w = 0.8 / k
        top = 0
        for i, (label, key, scale) in enumerate(series):
            ys = [num(r, key, scale) for r in rows]
            pos = [x - 0.4 + w * (i + 0.5) for x in xs]
            ax.bar(pos, [y or 0 for y in ys], width=w, color=SERIES[i], edgecolor=SURFACE, linewidth=2, label=label)
            for x, y in zip(pos, ys):
                if y is not None:
                    ax.annotate(fmt(y), (x, y), xytext=(0, 2), textcoords="offset points", ha="center",
                                va="bottom", fontsize=7, color=INK_2)
            top = max([top] + [y for y in ys if y is not None])
        if ref and ref[1] is not None:
            ax.axhline(ref[1], color=INK_2, linestyle="--", linewidth=1)
            ax.annotate(f"{ref[0]} {fmt(ref[1])}", (len(rows) - 0.45, ref[1]), xytext=(0, 3),
                        textcoords="offset points", ha="right", fontsize=8, color=INK_2)
            top = max(top, ref[1])
        ax.set_ylim(0, top * 1.2 if top else 1)
        ax.set_title(title, pad=22)
        legend_above(ax)

    ax = axes[0, 0]
    ok = [num(r, "tasks.ok") or 0 for r in rows]
    bad = [(num(r, "tasks.n") or 0) - o for r, o in zip(rows, ok)]
    ax.bar(xs, ok, width=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2, label="completed")
    ax.bar(xs, bad, bottom=ok, width=0.55, color=SERIES[1], edgecolor=SURFACE, linewidth=2, label="failed / timeout")
    for x, o, b in zip(xs, ok, bad):
        ax.annotate(f"{o:.0f}/{o + b:.0f}", (x, o + b), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_ylim(0, max(o + b for o, b in zip(ok, bad)) * 1.2 or 1)
    ax.set_title("Tasks completed of offered", pad=22)
    legend_above(ax)

    bars(axes[0, 1], "Throughput: completed tasks per minute", "throughput.tasks_per_min", "{:.3f}".format)
    paired(axes[0, 2], "Task completion time (arrive → end), s",
           [("p50", "completion_s.p50", 1), ("p95", "completion_s.p95", 1)], "{:.0f}".format)
    paired(axes[0, 3], "Queueing time (arrive → start), s",
           [("p50", "queue_s.p50", 1), ("p95", "queue_s.p95", 1)], "{:.0f}".format)
    paired(axes[1, 0], "Server memory (peak), MiB",
           [("RSS", "memory.server_rss_mib.peak", 1), ("anon (heap, incl. KV)", "memory.server_anon_mib.peak", 1)],
           "{:.0f}".format)
    n_ctx = next((num(r, "kv_cache.n_ctx") for r in rows if num(r, "kv_cache.n_ctx")), None)
    paired(axes[1, 1], "Context tokens (peak) vs KV capacity",
           [("server slots", "memory.slot_context.slot_context_tokens_total_peak", 1),
            ("running tasks", "memory.ctx_tokens_held.peak", 1)], "{:.0f}".format,
           ref=("n_ctx", n_ctx) if len({num(r, "kv_cache.n_ctx") for r in rows}) == 1 else None)

    ax = axes[1, 2]
    parts = [("prefill", "requests.prefill_ms.mean", 1e-3), ("decode", "requests.decode_ms.mean", 1e-3),
             ("waiting in server", "requests.non_compute_s.mean", 1)]
    bottom = [0.0] * len(rows)
    for i, (label, key, scale) in enumerate(parts):
        ys = [num(r, key, scale) or 0 for r in rows]
        ax.bar(xs, ys, bottom=bottom, width=0.55, color=SERIES[i], edgecolor=SURFACE, linewidth=2, label=label)
        bottom = [b + y for b, y in zip(bottom, ys)]
    for x, r, total in zip(xs, rows, bottom):       # decode share: the number the bottleneck argument needs
        p, d = num(r, "requests.prefill_ms.mean"), num(r, "requests.decode_ms.mean")
        share = f"\ndecode {100 * d / (p + d):.0f}%" if p is not None and d and len(rows) <= 8 else ""
        ax.annotate(f"{total:.0f} s{share}", (x, total), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_ylim(0, max(bottom) * 1.28 or 1)
    ax.set_title("Mean time per LLM request, s", pad=22)
    legend_above(ax)

    paired(axes[1, 3], "Prompt tokens served from KV cache, %",
           [("first step", "requests.cache_hit_ratio.first_step", 100),
            ("later steps", "requests.cache_hit_ratio.later_steps", 100)], "{:.0f}".format)
    axes[1, 3].set_ylim(0, 115)

    def setting(label, key):
        values = {r.get(key) or "n/a" for r in rows}
        return f"{label} {values.pop() if len(values) == 1 else 'varies'}"
    cond = ", ".join(setting(*s) for s in [("C", "concurrency"), ("tasks/run", "n_tasks"), ("ctx", "ctx_size"),
                                           ("slots", "slots"), ("threads", "threads")])
    fig.suptitle(f"{title} — {cond}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"group curves -> {path}")


# ---------------------------------------------------------------------------- per-run timeline

def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def plot_timeline(run, path, title):
    ev = read_jsonl(run / "events.jsonl")
    mem = read_jsonl(run / "memory.jsonl")
    cfg = json.loads((run / "config.json").read_text())
    n_ctx = (cfg.get("server") or {}).get("n_ctx")
    if not ev:
        return
    order = [e["task_id"] for e in ev if e["event"] == "task_arrive"]
    first = {}
    spans = {t: [] for t in order}                  # (start, end, kind)
    for e in ev:
        t, tid = e["t"] / 60, e.get("task_id")
        if e["event"] == "task_arrive":
            first[tid] = t
        elif e["event"] == "task_start":
            spans[tid].append((first[tid], t, "queue"))
        elif e["event"] in ("llm_start", "tool_start"):
            first[(tid, e["event"].split("_")[0])] = t
        elif e["event"] in ("llm_end", "tool_end"):
            kind = e["event"].split("_")[0]
            start = first.pop((tid, kind), None)
            if start is not None:
                spans[tid].append((start, t, kind))
    end = max(e["t"] for e in ev) / 60

    style()
    fig, axes = plt.subplots(3, 1, figsize=(13, 3 + 0.35 * len(order) + 5), sharex=True,
                             gridspec_kw=dict(height_ratios=[max(2, 0.35 * len(order)), 2.2, 2.2]))
    ax = axes[0]
    tidy(ax, "x")
    colors = dict(queue=QUEUE, llm=SERIES[0], tool=SERIES[1])
    names = dict(queue="waiting in FIFO queue", llm="LLM request", tool="tool (idle)")
    seen = set()
    for row, tid in enumerate(order):
        for s, e, kind in spans[tid]:
            ax.barh(row, max(e - s, 1e-3), left=s, height=0.62, color=colors[kind], edgecolor=SURFACE,
                    linewidth=1, label=None if kind in seen else names[kind])
            seen.add(kind)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_title("Tasks in arrival order", pad=22)
    legend_above(ax)

    ts = [m["t"] / 60 for m in mem]
    ax = axes[1]
    tidy(ax)
    total = [m.get("slot_context_tokens_total") for m in mem]
    busy = [m.get("slot_context_tokens_busy") for m in mem]
    if any(v is not None for v in total):
        ax.plot(ts, total, color=SERIES[0], linewidth=2, label="all slots (incl. idle retained)")
        ax.plot(ts, busy, color=SERIES[1], linewidth=2, label="busy slots")
    if n_ctx:
        ax.axhline(n_ctx, color=INK_2, linestyle="--", linewidth=1)
        ax.annotate(f"n_ctx {n_ctx}", (end, n_ctx), xytext=(0, 3), textcoords="offset points", ha="right",
                    fontsize=8, color=INK_2)
    ax.set_ylim(0, max([n_ctx or 0] + [v for v in total if v is not None]) * 1.15 or 1)
    ax.set_title("Slot context tokens (/slots; a token count, not KV bytes)", pad=22)
    legend_above(ax)

    ax = axes[2]
    tidy(ax)
    rss = [m.get("rss_mb") for m in mem]
    anon = [m.get("srv_anon_mib") for m in mem]
    if any(v is not None for v in rss):
        ax.plot(ts, rss, color=SERIES[0], linewidth=2, label="server RSS")
    if any(v is not None for v in anon):
        ax.plot(ts, anon, color=SERIES[1], linewidth=2, label="server anon (heap, incl. KV)")
    vals = [v for v in rss + anon if v is not None]
    ax.set_ylim(0, max(vals) * 1.15 if vals else 1)
    ax.set_title("Server memory, MiB", pad=22)
    ax.set_xlabel("time since run start, min")
    if vals:
        legend_above(ax)
    ax.set_xlim(0, end * 1.01)

    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", type=Path, help="results/<group>-<TS> folders holding summary_all.csv")
    ap.add_argument("--no-timeline", action="store_true", help="only curves_group.png")
    args = ap.parse_args()
    for group in args.groups:
        summary = group / "summary_all.csv"
        if not summary.exists():
            print(f"skip {group}: no summary_all.csv (run the group script first)")
            continue
        with summary.open(newline="") as f:
            rows = sorted(csv.DictReader(f), key=lambda r: natural(r["experiment"]))
        if not rows:
            print(f"skip {group}: empty summary_all.csv")
            continue
        title = group.name + (" (fake LLM)" if all(r.get("test") == "True" for r in rows) else "")
        plot_group(rows, group / "curves_group.png", title)
        if args.no_timeline:
            continue
        (group / "timelines").mkdir(exist_ok=True)
        for r in rows:
            run = group / r["run"]
            plot_timeline(run, group / "timelines" / f"{r['experiment']}.png", f"{title} / {r['run']}")
        print(f"timelines -> {group / 'timelines'}/")


if __name__ == "__main__":
    main()
