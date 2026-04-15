# tyclone

[![build](https://github.com/kojix2/tyclone/actions/workflows/build.yml/badge.svg)](https://github.com/kojix2/tyclone/actions/workflows/build.yml)
[![Lines of Code](https://img.shields.io/endpoint?url=https%3A%2F%2Ftokei.kojix2.net%2Fbadge%2Fgithub%2Fkojix2%2Ftyclone%2Flines)](https://tokei.kojix2.net/github/kojix2/tyclone)

tyclone is an unofficial reimplementation of both [PyClone-VI](https://github.com/Roth-Lab/pyclone-vi) and [PyClone](https://github.com/Roth-Lab/pyclone).
Written in Crystal CLI with a Rust kernel.

## Scope

- `fit-vi`: PyClone-VI style variational inference
- `fit-mcmc`: PyClone-inspired MCMC inference
- TSV input with PyClone-VI-compatible core fields
- deterministic runs with fixed seeds
- Rust-side parallelism with `--kernel-threads`

## Build

Requirements:

- Crystal
- Rust / Cargo
- make

Main workflows are exposed through the Makefile.

```bash
make build
```

The resulting binary is `bin/tyclone`.

## Test

```bash
make test
```

## Run

Variational inference:

```bash
./bin/tyclone fit-vi -i ../pyclone-vi/examples/synthetic.tsv -o out.tsv
```

Deterministic VI run:

```bash
./bin/tyclone fit-vi -i ../pyclone-vi/examples/synthetic.tsv -o out.tsv -c 4 -d beta-binomial -g 21 -r 2 --max-iters=200 --precision=1000 --seed=7 --kernel-threads=1 --restart-parallelism=1 --print-freq=0
```

MCMC run:

```bash
./bin/tyclone fit-mcmc -i input.tsv -o out.tsv -c 10 -d beta-binomial --num-iters=1000 --burnin=0 --thin=1 --precision=200 --seed=7 --print-freq=0
```

Expected input columns are:

- `mutation_id`
- `sample_id`
- `ref_counts`
- `alt_counts`
- `major_cn`
- `minor_cn`
- `normal_cn`

Optional columns:

- `tumour_content` default `1.0`
- `error_rate` default `0.001`

Larger VI run:

```bash
./bin/tyclone fit-vi -i ../pyclone-vi/examples/tracerx.tsv -o out.tsv -c 40 -d beta-binomial -r 2 --precision=200 --seed=7 --print-freq=0
```

## Common Options

- `-i`, `--in-file`: input TSV
- `-o`, `--out-file`: output TSV
- `-c`, `--num-clusters`: cluster cap
- `-d`, `--density`: `binomial` or `beta-binomial`
- `--seed`: fixed seed for reproducibility
- `--print-freq`: progress output frequency

VI-only:

- `-g`, `--num-grid-points`: CCF grid size
- `-r`, `--num-restarts`: number of restarts
- `--max-iters`: maximum VI iterations
- `--mix-weight-prior`: Dirichlet prior weight
- `--precision`: beta-binomial precision
- `--kernel-threads`: Rust kernel parallelism
- `--restart-parallelism`: outer restart parallelism

MCMC-only:

- `--num-iters`: total iterations before burn-in / thinning
- `--burnin`: number of saved samples to discard
- `--thin`: keep every N-th saved sample
- `--alpha`: CRP concentration
- `--init-method`: `connected` or `disconnected`

## Diagnostics

Restart diagnostics for VI:

```bash
PCV_DEBUG_RESTART_METRICS_FILE=restart_metrics.csv \
./bin/tyclone fit-vi -i ../pyclone-vi/examples/tracerx.tsv -o out.tsv -c 40 -d beta-binomial -r 2 --precision=200 --seed=7 --print-freq=1
```

This writes one row per restart with:

- `restart`
- `seed`
- `final_elbo`
- `used_clusters`
- `is_best`

Optional kernel profiling:

```bash
PCV_PROFILE=1 \
./bin/tyclone fit-vi -i ../pyclone-vi/examples/tracerx.tsv -o out.tsv -c 4 -d beta-binomial -r 1 --precision=200 --seed=7 --print-freq=0
```

This prints aggregated timings to stderr for:

- initial ELBO
- `update_z`
- `update_pi`
- `update_theta`
- iterative ELBO recomputation

## Current status

- Crystal CLI and Rust kernel are wired end to end
- VI and MCMC entry points are both available
- VI restart selection uses best ELBO
- Rust hot paths use Rayon when enabled
- output rows are inference-derived and cluster IDs are compactly renumbered
- tests cover Rust units, Crystal specs, and a deterministic golden output check

## Attribution And License

This project is a close reimplementation developed with direct reference to the upstream projects, their published methods, and their user-facing behavior.

- Upstream projects: PyClone, PyClone-VI
- Upstream repositories: Roth-Lab/pyclone, Roth-Lab/pyclone-vi

The upstream projects are distributed under GNU GPL v3 or later. This project is distributed under GPL v3 or later as well.
