# tyclone

[![build](https://github.com/kojix2/tyclone/actions/workflows/build.yml/badge.svg)](https://github.com/kojix2/tyclone/actions/workflows/build.yml)
[![Lines of Code](https://img.shields.io/endpoint?url=https%3A%2F%2Ftokei.kojix2.net%2Fbadge%2Fgithub%2Fkojix2%2Ftyclone%2Flines)](https://tokei.kojix2.net/github/kojix2/tyclone)
![Static Badge](https://img.shields.io/badge/PURE-VIBE_CODING-magenta)
[![DOI](https://zenodo.org/badge/1207957296.svg)](https://doi.org/10.5281/zenodo.20711091)

An unofficial PyClone and PhyClone Clone for Clonal Analysis

tyclone is an unofficial reimplementation of [PyClone-VI](https://github.com/Roth-Lab/pyclone-vi) and [PhyClone](https://github.com/Roth-Lab/phyclone).
Written in Crystal CLI with a Rust kernel.

## Scope

- `vi`: PyClone-VI style variational inference
- `phy run`: PhyClone-style tree trace generation (JSONL)
- `phy map`: MAP-like summary from phy trace
- `phy consensus`: topology + clade consensus summary
- `phy topology-report`: topology support summary from phy trace
- TSV input with PyClone-VI-compatible core fields
- deterministic runs with fixed seeds
- Rust-side parallelism with `--kernel-threads`

## Mode maturity

- PyClone-VI mode is near-parity with the original PyClone-VI implementation.
- PhyClone mode is under active parity development against upstream PhyClone.
- Strict PhyClone posterior, sampler, and output parity are still in progress.

## Build

Requirements:

- Crystal
- Rust / Cargo
- make

Main workflows are exposed through the Makefile.

```bash
make build
```

For an optimized Crystal CLI build, use:

```bash
make build release=1
```

The resulting binary is `bin/tyclone`.

## Test

```bash
make test
```

## Run

Variational inference:

```bash
./bin/tyclone vi -i ../pyclone-vi/examples/synthetic.tsv -o out.tsv
```

Deterministic VI run:

```bash
./bin/tyclone vi -i ../pyclone-vi/examples/synthetic.tsv -o out.tsv -c 4 -d beta-binomial -g 21 -r 2 --max-iters=200 --precision=1000 --seed=7 --kernel-threads=1 --restart-parallelism=1 --print-freq=0
```

Phy workflow:

```bash
./bin/tyclone phy run -i input.tsv -o trace.jsonl --num-iters=50 --num-chains=2 --num-particles=16 --burnin=1000 --seed=7
./bin/tyclone phy map -i trace.jsonl -o map.json
./bin/tyclone phy consensus -i trace.jsonl -o consensus.json --consensus-threshold=0.5
./bin/tyclone phy topology-report -i trace.jsonl -o topology_report.json
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
./bin/tyclone vi -i ../pyclone-vi/examples/tracerx.tsv -o out.tsv -c 40 -d beta-binomial -r 2 --precision=200 --seed=7 --print-freq=0
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
- `--debug-init-file`: debug-only JSON file with `pi`, `theta`, `z` arrays for same-initial-state validation

Phy run only:

- `-b`, `--burnin`: burn-in iterations
- `--num-iters`: main-chain MCMC iterations
- `--num-chains`: number of chains
- `--num-particles`: particle count
- `--thin`: trace thinning interval
- `--resample-threshold`: SMC ESS resampling threshold
- `-p`, `--proposal`: `bootstrap`, `fully-adapted`, or `semi-adapted`
- `-s`, `--subtree-update-prob`: subtree PG probability
- `--num-samples-data-point`: data-point Gibbs passes per iteration
- `--num-samples-prune-regraph`: prune-regraph passes per iteration
- `--concentration-update`, `--no-concentration-update`: concentration update toggle
- `--concentration-value`: initial concentration value
- `--grid-size`: CCF grid size for the exact outlier model
- `-l`, `--outlier-prob`: fallback outlier prior
- `-t`, `--max-time`: maximum runtime in seconds
- `-c`, `--cluster-file`: optional cluster assignment TSV

Python helper selection:

- `TYCLONE_PYTHON`: default Python executable for helper scripts

Notes:

- `vi --python-compatible` still expects a Python 3 executable with NumPy support for `default_rng`

## Diagnostics

Restart diagnostics for VI:

```bash
PCV_DEBUG_RESTART_METRICS_FILE=restart_metrics.csv \
./bin/tyclone vi -i ../pyclone-vi/examples/tracerx.tsv -o out.tsv -c 40 -d beta-binomial -r 2 --precision=200 --seed=7 --print-freq=1
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
./bin/tyclone vi -i ../pyclone-vi/examples/tracerx.tsv -o out.tsv -c 4 -d beta-binomial -r 1 --precision=200 --seed=7 --print-freq=0
```

This prints aggregated timings to stderr for:

- initial ELBO
- `update_z`
- `update_pi`
- `update_theta`
- iterative ELBO recomputation

Debug-only initial value injection:

```bash
./bin/tyclone vi -i ../pyclone-vi/examples/synthetic.tsv -o out.tsv -c 4 -g 21 -r 1 --debug-init-file=init.json --print-freq=0
```

The JSON file must contain flat `pi`, `theta`, and `z` arrays matching:

- `pi`: `num_clusters`
- `theta`: `num_clusters * num_samples * num_grid_points`
- `z`: `num_mutations * num_clusters`

This hook is intended for implementation comparison and fairness checks, not normal runs.

## Current status

- Crystal CLI and Rust kernel are wired end to end
- VI entry point is available and near-parity with the original PyClone-VI implementation
- `phy run/map/consensus/topology-report` workflow is available with JSONL/JSON outputs
- VI restart selection uses best ELBO
- Rust hot paths use Rayon when enabled
- output rows are inference-derived and cluster IDs are compactly renumbered
- tests cover Rust units, Crystal specs, and a deterministic golden output check

## PhyClone parity notes

- The `phy` implementation is being aligned with upstream-compatible internals under `rust-kernel/src/phyclone/compat`, and `phy run` executes the compat Particle Gibbs / SMC sampler there.
- In `phy run`, `num_iters` is total iterations and recorded trace length follows post-`--burnin` / `--thin`.
- consensus output includes clade support and a `consensus_tree` reconstruction, while representative topology fields remain for compatibility.
- loss prior supports `cellular_prevalence`-informed assignment via `--cluster-file` metadata.
- trace and post-process outputs are currently JSONL/JSON; HDF5/archive parity is planned but not complete.

## Attribution And License

tyclone is an unofficial reimplementation of the methods below. Cite the original papers, not tyclone.

- PyClone-VI — [Roth-Lab/pyclone-vi](https://github.com/Roth-Lab/pyclone-vi) —
  Gillis & Roth, *BMC Bioinformatics* 2020. doi:[10.1186/s12859-020-03919-2](https://doi.org/10.1186/s12859-020-03919-2)
- PhyClone — [Roth-Lab/PhyClone](https://github.com/Roth-Lab/PhyClone) —
  Hurtado, Bouchard-Côté & Roth, *Bioinformatics* 2025. doi:[10.1093/bioinformatics/btaf344](https://doi.org/10.1093/bioinformatics/btaf344)
- PyClone — [Roth-Lab/pyclone](https://github.com/Roth-Lab/pyclone) —
  Roth et al., *Nature Methods* 2014. doi:[10.1038/nmeth.2883](https://doi.org/10.1038/nmeth.2883)

Upstream is GPL v3 or later; tyclone is GPL v3 or later as well.
