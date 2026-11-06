# XPBD scatter, gather, and coloring benchmark

The benchmark compares CUDA solvers from identical predicted positions, stopping
at the same relative RMS distance-constraint error. It measures projection only,
not simulation throughput or equal-iteration cost.

## Methods

- `scatter`: frozen-position Jacobi. Clear a particle correction buffer, solve
  every constraint once, atomically add endpoint corrections, then apply them in
  a particle pass.
- `gather`: frozen-position Jacobi. Solve every constraint once into a private
  per-constraint vector, then use a particle-to-constraint CSR adjacency to gather
  and apply corrections. This avoids duplicate constraint evaluation, at the cost
  of an intermediate array and indirect reads. It is not the alternative where
  each particle recomputes each incident constraint independently.
- `coloring`: greedy constraint coloring. Sequential color passes; independent
  constraints in a color directly update positions without atomics. This is
  multicolor nonlinear Gauss-Seidel and has different convergence from Jacobi.

All methods use the same distance gradients and inverse masses, FP32 positions,
and no fast math. Jacobi relaxes both position and multiplier increments by the
same scalar. The timed scenes use zero compliance (the hard-constraint limit of
XPBD). The scalar kernel includes the accumulated multiplier term for positive
compliance; that case is validated but not performance-tested here. The existing
viewer/core solver is unchanged.

## Scenes and error

Square regular cloth grids include horizontal/vertical structural links, both
diagonal shear links, and two-hop horizontal/vertical distance-bending links,
following the link families in Ten Minute Physics #16. All are hard in this
benchmark. Both top corners are fixed. The topology is identical between methods,
but differs from dynamics' triangulated viewer cloth and from TMP's hybrid
schedule. Spacing and free inverse mass are 1. There is no contact or gravity
during the projection solve.

- `jitter`: seed-42 independent uniform perturbations in [-0.08, 0.08] on each
  free particle coordinate.
- `strain`: smooth 4% vertical stretch, horizontal shear increasing with height,
  and an out-of-plane sinusoidal displacement. Pins retain their rest positions.

For each link, C = length - rest length. Error is
`sqrt(mean((C / rest_length)^2))`; targets are 0.01, 0.001, and 0.0001
(1%, 0.1%, 0.01%). Also record maximum absolute relative error; an RMS target is
not a per-link bound. For compliant constraints the implemented diagnostic uses
`(C + alpha_tilde * lambda) / rest_length`; this equals geometric error only in
the hard scenes used here.

Discovery measures error after every sweep and records the first crossing.
Unreached targets are not assigned a time. Discovery and reductions are excluded
from the timed replay; this measures the solve cost for a known work budget,
not an adaptive production stopping test. Every replay starts from the same
positions with multipliers reset, and its achieved error is checked afterwards.
Analysis excludes parameter groups if any timed replay exceeds its target.

## Timing

Measure CUDA events around the complete solve and host wall time including
submission and final synchronization. Clearing scatter's correction buffer,
gather's intermediate writes and adjacency traversal, and every coloring pass
are inside the timing. Allocations, preprocessing, initial upload/reset, error
checks, and rendering are excluded. CPU adjacency construction and coloring
times are reported separately; fixed-topology simulations can amortize them.
Their GPU upload cost is not measured.

Two launch modes are measured: ordinary kernel launches, and a CUDA graph of one
complete sweep replayed for every sweep. The latter retains synchronization
between color passes while reducing host submission overhead. It does not
capture the entire solve into one graph.

Nine trials per parameter/target/launch are measured, plus one discarded trial.
Jobs are shuffled with seed 123 after convergence discovery has warmed the GPU.
Report medians and min/max. Jacobi relaxation candidates are 0.20, 0.25, 0.30,
0.35, 0.40. Select the fastest median independently for each target and launch
mode, retaining all raw results. This tuning cost is excluded; this is an
optimistically tuned comparison, not a universally chosen relaxation value.
Pilot sweeps also tried 0.6, 0.8, 1.0 and found them divergent.

The GPU is shared with existing desktop, viewer, and compute applications. No
other process was stopped and clocks were not locked. Results describe this
session under background load, not an isolated hardware benchmark. Compare
effect sizes against trial spread.

## Build and run on blacknoir

```sh
cd ~/Projects/dynamics/benchmarks/xpbd_scatter_gather
nvcc -O3 -std=c++17 -arch=sm_89 -ccbin /usr/bin/g++-13 \
  --default-stream per-thread --fmad=false -lineinfo benchmark.cu -o benchmark
./benchmark --test
compute-sanitizer --tool memcheck --error-exitcode 1 ./benchmark --test
compute-sanitizer --tool racecheck --error-exitcode 1 ./benchmark --test
mkdir -p results
./benchmark 32 jitter 8192 results/32-jitter 9
./benchmark 128 jitter 8192 results/128-jitter 9
./benchmark 256 jitter 8192 results/256-jitter 9
./benchmark 32 strain 16384 results/32-strain 9
# Analysis script and historical report are preserved outside this repo:
# ~/Projects/dynamics-archive-2026-10-30/imported-references-2026-10-30/
```

CUDA 12.4, RTX 4090 (sm_89), driver 595.71.05. Tests check an analytic unequal-mass
compliant spring, mixed-compliance GPU/CPU iteration parity for all methods,
valid coloring, unchanged pins, and CUDA graph/ordinary-launch parity.

Ten Minute Physics #16 defaults to four independent structural-link passes plus
a Jacobi scatter pass for shear and bending, with Jacobi scale 0.2. That hybrid
is not identical to the fully colored method benchmarked here:
https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/16-GPUCloth.py
