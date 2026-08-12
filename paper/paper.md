---
title: "unclone: Performance and Numerical Agreement with PyClone-VI"
author: "kojix2"
date: "2026-08-12"
lang: en
documentclass: article
papersize: a4
geometry: margin=18mm
fontsize: 11pt
linestretch: 1.15
bibliography: paper.bib
---

# Abstract

We evaluated unclone [@unclone_github; @unclone_zenodo], a Rust-kernel reimplementation of PyClone-VI [@gillis2020pyclonevi], on a local 8-core/16-thread AMD Ryzen 7 5700X workstation. With NumPy-generated initial values shared between implementations, all 18 dataset--seed combinations produced identical mutation partitions (adjusted Rand index 1.0); the largest absolute difference in cellular prevalence (CCF) was $5.5\times10^{-13}$. At one thread, unclone was 2.8--5.9 times faster across inputs of 100--2,440 mutations, even though its measurement included CLI startup while PyClone-VI was measured in a warmed Python process. On the full TRACERx input, the unclone kernel accelerated by 4.80 times on eight physical cores and 5.10 times on 16 hardware threads. PyClone-VI improved by at most 1.07 times. At eight threads, full-input latency was 0.084 s for the unclone CLI and 1.723 s for warmed in-process PyClone-VI. Median peak resident memory was 33.4 MB and 280.1 MB, respectively.

# Methods

## Implementations and data

unclone was compiled with `make build release=1 cpu=native`; its base commit and the SHA-256 of the measured Rust source diff are stored in `data/environment.json`. PyClone-VI 0.2.0 was installed from upstream commit `07306831a9a4`. We used the official synthetic and TRACERx example files. PyClone-VI removes mutations not observed in every sample; therefore, the TRACERx timing input was filtered once to the 2,440 complete mutations before being passed to either implementation. Scaling inputs contain the first 100, 250, 500, 1,000, or all 2,440 complete mutations. SHA-256 hashes of every generated input are also stored in `data/environment.json`.

The measured unclone kernel uses the same Rayon execution path at every thread count. Variational contractions and the initial expected log likelihood share matrix-multiplication kernels, while beta-binomial terms constant across the CCF grid are evaluated once per observation.

Both implementations used beta-binomial density, 40 clusters, 100 CCF grid points, precision 200, convergence tolerance $10^{-6}$, at most 10,000 iterations, one restart, and seed 7. Numerical comparisons additionally used seeds 42 and 123 and unclone's `--python-compatible` initialization.

## Timing protocol

Numba was compiled by an unrecorded warm-up before PyClone-VI timing. PyClone-VI was then measured in one persistent process; its timer includes input loading, likelihood construction, initialization, and inference, but excludes result serialization. BLAS and Numba thread counts were both set to the requested value. unclone was launched as a fresh CLI process on every repetition. Cross-tool latency uses its complete CLI wall time, including input and output. Kernel-only time reported by the Rust profiling boundary is used for parallel speedup, so fixed CLI overhead does not obscure scaling.

Each timing condition was repeated seven times in a deterministic randomized order. We report medians and interquartile ranges (IQR). Runs using one through eight threads were pinned to distinct physical cores 0 through $n-1$; the 16-thread condition used both hardware threads of all eight cores. Peak resident set size (RSS) was measured in three fresh processes with GNU `time`. Raw observations, summaries, and the executable benchmark driver are retained under `paper/data` and `paper/scripts/benchmark.py`.

## Computing environment

  Item                  Value
  --------------------- ------------------------------------------
  CPU                   AMD Ryzen 7 5700X
  Topology              8 cores, 16 hardware threads, one socket
  Memory                60 GiB
  OS                    Linux 7.0.0-29-generic, x86-64
  CPU governor          `powersave` (`amd-pstate-epp`), boost enabled
  unclone               0.0.4, native CPU build
  PyClone-VI            0.2.0
  Rust / Crystal        1.97.1 / 1.21.0
  Python                3.14.4
  NumPy / Numba / SciPy 2.5.2 / 0.67.0 / 1.18.0

Table: Measurement environment. The host exposed no hypervisor and was otherwise idle; CPU frequency was not fixed.

# Numerical agreement

Across six input sizes and three seeds, matched initialization gave ARI 1.0 in every run. CCF correlation rounded to 1.0 throughout, and maximum absolute CCF differences remained between $1.9\times10^{-14}$ and $5.4\times10^{-13}$ (Table 2). Thus the tested implementation is numerically concordant with PyClone-VI at near-floating-point precision, although this is not a claim of bitwise identity for every platform or input.

  Dataset      Mutations  Seeds tested  Minimum ARI  Maximum $|\Delta\mathrm{CCF}|$
  ------------ ---------- ------------- ------------ ------------------------------
  synthetic    100        7, 42, 123    1.0          $1.3\times10^{-14}$
  TRACERx      100        7, 42, 123    1.0          $1.3\times10^{-14}$
  TRACERx      250        7, 42, 123    1.0          $1.2\times10^{-13}$
  TRACERx      500        7, 42, 123    1.0          $1.5\times10^{-13}$
  TRACERx      1,000      7, 42, 123    1.0          $3.8\times10^{-13}$
  TRACERx      2,440      7, 42, 123    1.0          $5.5\times10^{-13}$

Table: Agreement with shared NumPy initialization (`data/quality.csv`).

# Performance

## Single-thread comparison

At one thread, unclone was faster for every input despite the conservative timing boundary: unclone includes process startup and serialization, whereas PyClone-VI excludes process startup, JIT compilation, and serialization. Median speedups ranged from 2.81 to 5.88 times (Figure 1 and Table 3). On full TRACERx, median latency was 0.313 s versus 1.838 s. IQR was 0.024 s and 0.012 s, respectively.

  Dataset      Mutations  U CLI (s)  PCV (s)  PCV/U
  ------------ ---------- ---------- -------- ----------
  synthetic    100        0.023      0.092    3.99
  TRACERx      100        0.027      0.076    2.81
  TRACERx      250        0.038      0.177    4.60
  TRACERx      500        0.078      0.346    4.43
  TRACERx      1,000      0.170      0.719    4.24
  TRACERx      2,440      0.313      1.838    5.88

Table: One-thread median latency over seven runs (`data/timing_raw.csv`). U = unclone; PCV = PyClone-VI; IP = warmed in-process measurement.

![One-thread latency and speedup.](figures/fig1_performance.png){width=100%}

## Multithread scaling

The full TRACERx fit shows effective Rust-kernel parallelism (Figure 2). Relative to one thread, unclone achieved speedups of 1.52, 3.16, 4.80, and 5.10 times at 2, 4, 8, and 16 threads. Eight-core parallel efficiency was 60%. Simultaneous multithreading reduced 0.061 s at eight threads to 0.057 s at 16 threads, only a further 6% improvement, indicating saturation near the number of physical cores. The lower relative speedup than in the previous implementation follows a 56% reduction in the one-thread kernel time; absolute multithread latency also improved.

PyClone-VI stayed between 1.71 and 1.84 s. Its best observed median was at four threads, only 1.07 times faster than one thread; the 16-thread result regressed to 1.81 s. The result applies to this workload and linked OpenBLAS/Numba versions, rather than to all possible PyClone-VI inputs.

\clearpage

  Threads  U kernel (s)  U speedup  PCV (s)  PCV speedup
  -------- ------------- ---------- -------- ------------
  1        0.293         1.00       1.838    1.00
  2        0.193         1.52       1.753    1.05
  4        0.093         3.16       1.714    1.07
  8        0.061         4.80       1.723    1.07
  16       0.057         5.10       1.808    1.02

Table: Thread scaling on 2,440 TRACERx mutations (`data/thread_summary.csv`). U = unclone kernel; PCV = warmed in-process PyClone-VI.

![Fit time and parallel speedup on full TRACERx.](figures/fig2_threads.png){width=100%}

## Memory

At one thread, median peak RSS was 32.7 MB for unclone and 277.0 MB for PyClone-VI, an 8.5-fold difference. At eight threads the values were 33.4 MB and 280.1 MB, an 8.4-fold difference. Additional worker threads therefore added little memory in either implementation on this input.

# Limitations

This study uses one machine, one compiler build, one upstream example family, and short-running workloads. Dynamic frequency scaling remained enabled; randomized ordering, CPU affinity, and repeated medians reduce but do not eliminate frequency and thermal effects. Small-input measurements have larger relative variation because they approach process-startup duration. The cross-tool latency comparison intentionally favors PyClone-VI by warming JIT and excluding its process startup; kernel parallel scaling uses narrower implementation-specific timing boundaries. Results should therefore be interpreted as reproducible measurements of this configuration, not universal hardware-independent constants.

# Conclusions

On a stable local 8-core workstation, unclone reproduced PyClone-VI partitions and CCF estimates to near machine precision, was 2.8--5.9 times faster at one thread, and used about one eighth of the process memory. Its Rust kernel reached 4.80-fold acceleration on eight physical cores, whereas PyClone-VI showed little thread scaling on the same workload. Hyperthreading provided only a modest additional gain. These measurements replace the earlier two-vCPU cloud results and directly characterize multithread performance.

# References
