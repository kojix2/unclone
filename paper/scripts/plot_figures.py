#!/usr/bin/env python3
"""Generate the paper's monochrome figures."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; DATA, OUT = ROOT / "data", ROOT / "figures"
BLACK, GREY, LIGHT = "0.1", "0.5", "0.85"


def read(name):
    with (DATA / name).open(newline="") as f: return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout(pad=.5); fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    timing = read("timing_summary.csv"); datasets = ["synthetic", "tracerx_100", "tracerx_250", "tracerx_500", "tracerx_1000", "tracerx_2440"]
    labels = ["syn\n100", "tx\n100", "tx\n250", "tx\n500", "tx\n1000", "tx\n2440"]

    # End-to-end scaling at one thread.
    def times(tool, threads=1):
        return [float(next(r["median_s"] for r in timing if r["tool"] == tool and r["dataset"] == d and int(r["threads"]) == threads)) for d in datasets]
    u, p, x = times("unclone"), times("pyclone-vi"), range(len(datasets))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    axes[0].plot(x, u, "o-", color=BLACK, label="unclone CLI"); axes[0].plot(x, p, "s--", color=GREY, label="PyClone-VI warm IP")
    axes[0].set(ylabel="Median time (s)", title="A. One-thread latency"); axes[0].legend(frameon=False, fontsize=7)
    axes[1].bar(x, [a/b for a, b in zip(p, u)], color=LIGHT, edgecolor=BLACK)
    axes[1].set(ylabel="unclone speedup (×)", title="B. Speedup over PyClone-VI")
    for ax in axes: ax.set_xticks(x, labels, fontsize=7); ax.grid(axis="y", color=".9")
    save(fig, "fig1_performance.png")

    # Thread scaling on full TRACERx.
    thread = read("thread_summary.csv"); ts = [1, 2, 4, 8, 16]
    series = {tool: [float(next(r["median_s"] for r in thread if r["tool"] == tool and int(r["threads"]) == t)) for t in ts] for tool in ["unclone", "pyclone-vi"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for tool, fmt, color in [("unclone", "o-", BLACK), ("pyclone-vi", "s--", GREY)]:
        axes[0].plot(ts, series[tool], fmt, color=color, label=tool); axes[1].plot(ts, [series[tool][0]/v for v in series[tool]], fmt, color=color, label=tool)
    axes[0].set(ylabel="Median time (s)", title="A. Threaded fit"); axes[1].set(ylabel="Speedup vs 1 thread (×)", title="B. Parallel speedup")
    axes[1].plot(ts, ts, ":", color=".7", label="ideal")
    for ax in axes: ax.set(xlabel="Threads", xticks=ts); ax.grid(axis="y", color=".9"); ax.legend(frameon=False, fontsize=7)
    save(fig, "fig2_threads.png")

if __name__ == "__main__": main()
