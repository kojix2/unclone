# toyclone

toyclone is a reimplementation of PyClone-VI.

It is built as a Crystal CLI frontend with a Rust kernel backend for practical
use and experimentation.

## What It Does

- reads PyClone-VI style TSV input
- runs variational inference in a Rust kernel
- writes PyClone-VI style TSV output
- supports restart selection by best ELBO
- supports Rayon-based parallel execution with `--kernel-threads`

## Build

Requirements:

- Crystal
- Rust / Cargo
- `make`

This project uses Makefile for all main workflows.

```bash
make build
```

The resulting `bin/toyclone` binary statically links the Rust kernel archive, so
it does not require `libpcv_kernel.so` at runtime.

## Test

```bash
make test
```

## Run

Basic run:

```bash
./bin/toyclone fit -i ../pyclone-vi/examples/synthetic.tsv -o out.tsv
```

Deterministic run with fixed settings:

```bash
./bin/toyclone fit -i ../pyclone-vi/examples/synthetic.tsv -o out.tsv -c 4 -g 21 -r 2 --max-iters=200 --seed=7 --print-freq=0
```

TRACERx-sized example:

```bash
./bin/toyclone fit -i ../pyclone-vi/examples/tracerx.tsv -o out.tsv -c 40 -d beta-binomial -r 2 --precision=200 --seed=7 --print-freq=0
```

`make run` remains available as a convenience wrapper, but it is no longer
needed to inject a runtime library path for the Rust kernel.

## Useful Options

- `-c`, `--num-clusters`: upper bound on cluster count
- `-d`, `--density`: `binomial` or `beta-binomial`
- `-g`, `--num-grid-points`: CCF grid size
- `-r`, `--num-restarts`: number of random restarts
- `--max-iters`: maximum variational iterations
- `--seed`: fixed seed for reproducibility
- `--kernel-threads`: Rayon thread count for kernel-side parallel work
- `--print-freq`: restart / progress diagnostics frequency

Restart diagnostics (for validation/debug):

```bash
PCV_DEBUG_RESTART_METRICS_FILE=restart_metrics.csv \
./bin/toyclone fit -i ../pyclone-vi/examples/tracerx.tsv -o out.tsv -c 40 -d beta-binomial -r 2 --precision=200 --seed=7 --print-freq=1
```

This writes a CSV with one row per restart:

- `restart`
- `seed`
- `final_elbo`
- `used_clusters`
- `is_best`

Optional profiling:

```bash
PCV_PROFILE=1 \
./bin/toyclone fit -i ../pyclone-vi/examples/tracerx.tsv -o out.tsv -c 4 -d beta-binomial -r 1 --precision=200 --seed=7 --print-freq=0
```

This prints aggregated kernel timing to stderr for:

- initial ELBO
- `update_z`
- `update_pi`
- `update_theta`
- iterative ELBO recomputation

## Current status

- Crystal CLI + Rust kernel end-to-end pipeline implemented
- Variational inference loop (`update_z -> update_pi -> update_theta`) implemented
- Restart selection by best ELBO implemented
- Restart-level parallelism implemented in Rust via `rayon` and controlled by `--kernel-threads`
- Likelihood tensor construction parallelism implemented in Rust via `rayon`
- Variational inference hot paths (`update_z`, `update_theta`, ELBO-side contractions) parallelized via `rayon` when `--kernel-threads > 1`
- OpenBLAS was evaluated experimentally, but the current supported parallel backend remains `rayon`
- Optional profiling is available with `PCV_PROFILE=1` to print per-phase timings to stderr
- Output is inference-derived (non-dummy)
- Cluster IDs are compactly renumbered across used clusters
- Test coverage includes Crystal specs, Rust unit tests, and a deterministic golden output check

## Attribution And License

This project is a close reimplementation of PyClone-VI developed with direct
reference to the upstream project, its published method, and its user-facing
behavior.

- Original project: PyClone-VI
- Upstream repository: Roth-Lab/pyclone-vi
- Paper: PyClone-VI: scalable inference of clonal population structures using whole genome data

The upstream PyClone-VI project is distributed under GNU GPL v3 or later. This
project is distributed under GPL v3 or later as well.
