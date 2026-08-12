#!/usr/bin/env python3
"""Reproduce the paper benchmarks (run from the repository root)."""
import argparse, contextlib, csv, hashlib, json, os, platform, random, re, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT, WORK = ROOT / "paper/data", Path("/tmp/unclone-paper-benchmark")
THREADS, REPS = [1, 2, 4, 8, 16], 7
PROFILE = re.compile(r"\[unclone-kernel-profile\].* total_ms=([0-9.]+)")


def affinity(n):
    return list(range(n if n <= 8 else 16))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)


def inputs(upstream):
    WORK.mkdir(exist_ok=True); paths = {"synthetic": upstream / "examples/synthetic.tsv"}
    with (upstream / "examples/tracerx.tsv").open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t")); fields = rows[0].keys()
    samples = {r["sample_id"] for r in rows}; count = {}
    for r in rows: count[r["mutation_id"]] = count.get(r["mutation_id"], 0) + 1
    rows = [r for r in rows if count[r["mutation_id"]] == len(samples)]
    mutations = list(dict.fromkeys(r["mutation_id"] for r in rows))
    for n in [100, 250, 500, 1000, len(mutations)]:
        keep = set(mutations[:n]); path = WORK / f"tracerx_{n}.tsv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fields, delimiter="\t"); w.writeheader(); w.writerows(r for r in rows if r["mutation_id"] in keep)
        paths[f"tracerx_{n}"] = path
    return paths


def pyclone_fit(path, threads=1, seed=7, restarts=1):
    import numpy as np
    from numba import set_num_threads
    from threadpoolctl import threadpool_limits
    from pyclone_vi.data import load_data
    from pyclone_vi.inference import DataPreprocessor, Priors, VariationalParameters, fit_pyclone_model
    os.sched_setaffinity(0, affinity(threads)); set_num_threads(threads)
    with threadpool_limits(limits=threads), open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
        start = time.perf_counter(); log_p, mutations, samples = load_data(path, "beta-binomial", 100, precision=200)
        priors, prep, rng, best = Priors(40, 100, 1.0), DataPreprocessor(log_p), np.random.default_rng(seed), None
        for _ in range(restarts):
            var = VariationalParameters(40, log_p.shape[0], log_p.shape[1], log_p.shape[2], rng)
            trace = fit_pyclone_model(priors, var, prep, 1e-6, 10000, 1000000)
            if best is None or trace[-1] > best[0]: best = (trace[-1], var)
        elapsed = time.perf_counter() - start
    return elapsed, (mutations, samples, best[1])


def unclone(path, threads=1, seed=7, restarts=1, restart_threads=1, compatible=False):
    output = WORK / "unclone.tsv"
    cmd = ["taskset", "-c", ",".join(map(str, affinity(max(threads, restart_threads)))), str(ROOT / "bin/unclone"), "vi",
           "-i", str(path), "-o", str(output), "-c", "40", "-d", "beta-binomial", "-g", "100", "-r", str(restarts),
           "--precision=200", f"--seed={seed}", f"--kernel-threads={threads}", f"--restart-parallelism={restart_threads}", "--print-freq=0"]
    if compatible: cmd.append("--python-compatible")
    env = os.environ | {"PCV_PROFILE": "1", "UNCLONE_PYTHON": sys.executable}
    start = time.perf_counter(); run = subprocess.run(cmd, text=True, capture_output=True, check=True, env=env); wall = time.perf_counter() - start
    return wall, float(PROFILE.search(run.stderr).group(1)) / 1000, output


def summarize(rows, keys):
    groups = {}
    for row in rows: groups.setdefault(tuple(row[k] for k in keys), []).append(float(row["seconds"]))
    out = []
    for group, values in groups.items():
        q = statistics.quantiles(values, n=4); mean = statistics.mean(values)
        out.append(dict(zip(keys, group)) | {"median_s": statistics.median(values), "iqr_s": q[2]-q[0],
                    "min_s": min(values), "max_s": max(values), "cv_pct": 100*statistics.stdev(values)/mean})
    return out


def timings(paths):
    pyclone_fit(paths["synthetic"])  # Compile Numba before any recorded run.
    cases = [(name, path, t) for name, path in paths.items() if name == "synthetic" or name.startswith("tracerx_")
             for t in ({1, 8} if name != "tracerx_2440" else set(THREADS))]
    jobs = [(tool, name, path, t, rep) for rep in range(REPS) for name, path, t in cases for tool in ["unclone", "pyclone-vi"]]
    random.Random(20260812).shuffle(jobs); rows = []
    for i, (tool, name, path, threads, rep) in enumerate(jobs, 1):
        if tool == "unclone": wall, seconds, _ = unclone(path, threads); metric = "kernel"
        else: seconds, _ = pyclone_fit(path, threads); wall, metric = seconds, "warm-in-process"
        rows.append({"tool": tool, "dataset": name, "mutations": 100 if name == "synthetic" else int(name.split("_")[1]),
                     "threads": threads, "rep": rep, "seconds": seconds, "wall_s": wall, "metric": metric})
        print(f"timing {i}/{len(jobs)} {tool} {name} t={threads}: {seconds:.3f}s", flush=True)
    write_csv(OUT / "timing_raw.csv", rows)
    comparison = [r | {"seconds": r["wall_s"] if r["tool"] == "unclone" else r["seconds"],
                       "metric": "cli-wall" if r["tool"] == "unclone" else r["metric"]} for r in rows]
    write_csv(OUT / "timing_summary.csv", summarize(comparison, ["tool", "dataset", "mutations", "threads", "metric"]))
    threading = [r for r in rows if r["dataset"] == "tracerx_2440"]
    write_csv(OUT / "thread_summary.csv", summarize(threading, ["tool", "threads", "metric"]))


def quality(paths):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    rows = []
    for name in ["synthetic", "tracerx_100", "tracerx_250", "tracerx_500", "tracerx_1000", "tracerx_2440"]:
        for seed in [7, 42, 123]:
            _, (mutations, samples, var) = pyclone_fit(paths[name], seed=seed)
            _, _, output = unclone(paths[name], seed=seed, compatible=True)
            with output.open(newline="") as f: u = {(r["mutation_id"], r["sample_id"]): r for r in csv.DictReader(f, delimiter="\t")}
            labels, grid = var.z.argmax(1), np.linspace(0, 1, var.theta.shape[2]); p_ccf, u_ccf = [], []
            u_labels = [int(u[(m, samples[0])]["cluster_id"]) for m in mutations]
            for i, mutation in enumerate(mutations):
                for j, sample in enumerate(samples):
                    p_ccf.append(float(grid @ var.theta[labels[i], j])); u_ccf.append(float(u[(mutation, sample)]["cellular_prevalence"]))
            delta = np.abs(np.array(p_ccf)-u_ccf)
            rows.append({"dataset": name, "mutations": len(mutations), "seed": seed, "ari": adjusted_rand_score(labels, u_labels),
                         "ccf_r": np.corrcoef(p_ccf, u_ccf)[0, 1], "ccf_max_abs": delta.max(), "ccf_mean_abs": delta.mean()})
            print("quality", name, seed, rows[-1], flush=True)
    write_csv(OUT / "quality.csv", rows)


def memory_worker(tool, path, threads):
    if tool == "pyclone-vi": pyclone_fit(path, threads)
    else: unclone(path, threads)


def memory(path):
    rows = []
    for tool in ["unclone", "pyclone-vi"]:
        for threads in [1, 8]:
            for rep in range(3):
                stamp = WORK / "rss.txt"; cmd = ["/usr/bin/time", "-f", "%M", "-o", str(stamp), sys.executable, __file__, "memory-worker", tool, str(path), str(threads)]
                subprocess.run(cmd, check=True, env=os.environ); rows.append({"tool": tool, "threads": threads, "rep": rep, "max_rss_mb": int(stamp.read_text())/1024})
    write_csv(OUT / "memory.csv", rows)


def environment(upstream, paths):
    import h5py, numba, numpy, pandas, scipy
    read = lambda p: Path(p).read_text().strip() if Path(p).exists() else "unknown"
    data = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "platform": platform.platform(), "python": platform.python_version(),
            "numpy": numpy.__version__, "numba": numba.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__, "h5py": h5py.__version__,
            "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(), "crystal": subprocess.check_output(["crystal", "--version"], text=True).splitlines()[0],
            "build": "make build release=1 cpu=native", "memory": subprocess.check_output(["free", "-h"], text=True),
            "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"), "affinity": "1-8 threads: CPUs 0..n-1; 16 threads: CPUs 0..15",
            "unclone_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "unclone_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--binary", "--", "rust-kernel"])).hexdigest(),
            "pyclone_vi_commit": subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
            "cpu": subprocess.check_output(["lscpu"], text=True), "inputs": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()}}
    (OUT / "environment.json").write_text(json.dumps(data, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(); p.add_argument("command", nargs="?", default="all"); p.add_argument("args", nargs="*"); p.add_argument("--upstream", type=Path, default=Path("/tmp/pyclone-vi-upstream")); a = p.parse_args()
    if a.command == "memory-worker": return memory_worker(a.args[0], Path(a.args[1]), int(a.args[2]))
    paths = inputs(a.upstream); environment(a.upstream, paths)
    if a.command in ("all", "timings"): timings(paths)
    if a.command in ("all", "quality"): quality(paths)
    if a.command in ("all", "memory"): memory(paths["tracerx_2440"])


if __name__ == "__main__": main()
