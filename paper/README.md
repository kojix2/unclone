# Performance paper

The paper has two reproducible steps:

```text
benchmark.py -> data/*.csv -> plot_figures.py -> figures/*.png
paper.md + paper.bib + figures/*.png -> Pandoc/XeLaTeX -> paper.pdf
```

## Build the PDF

Requires Python 3, matplotlib, Pandoc, and XeLaTeX.

```bash
cd paper
python3 -m pip install -r requirements.txt
make
```

GitHub Actions performs the same figure and PDF build. Every relevant push and
pull request gets an `unclone-paper-pdf` artifact; `v*` tags also attach the PDF
to the GitHub Release.

## Re-run the benchmarks

The recorded run used Linux on an AMD Ryzen 7 5700X. It requires GNU `time`,
`taskset`, Crystal, Rust, and Python 3.14. Build and install the pinned tools:

```bash
make build release=1 cpu=native
git clone https://github.com/Roth-Lab/pyclone-vi /tmp/pyclone-vi-upstream
git -C /tmp/pyclone-vi-upstream checkout 07306831a9a48275ccfe43fe42bbe18d7370bc72
python3 -m pip install -r paper/benchmark-requirements.txt
PYTHONPATH=/tmp/pyclone-vi-upstream python3 paper/scripts/benchmark.py
```

The driver filters the official TRACERx input to mutations observed in every
sample, pins worker threads to CPUs, warms Numba, randomizes 7 timing repetitions,
and writes raw observations plus summaries to `paper/data`. Inputs are generated
under `/tmp`; their SHA-256 hashes are recorded in `environment.json`.

Key files:

| Path | Purpose |
|---|---|
| `scripts/benchmark.py` | benchmark and agreement driver |
| `data/timing_raw.csv` | all repeated timing observations |
| `data/timing_summary.csv` | cross-tool median/IQR summaries |
| `data/thread_summary.csv` | full-input thread scaling |
| `data/quality.csv` | matched-initialization agreement |
| `data/memory.csv` | peak-RSS observations |
| `data/environment.json` | versions, commits, CPU, and input hashes |

This is a technical performance report, not a peer-reviewed publication.
