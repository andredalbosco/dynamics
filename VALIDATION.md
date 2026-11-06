# Validation

## Concrete spring energy terms — 1 November 2026

Added `SpringEnergy` with endpoints, rest length, and stiffness. Its `value`,
`vertexValue`, and `addDerivatives` methods own the elastic model; trial evaluation
uses a proposed vertex position without mutating particle state. Contributions
for unrelated vertices are zero. VBD now stores energy terms and owns inertia,
vertex assembly, the safeguarded solve, and line search. No virtual energy
hierarchy was introduced. Pendulum uses a shared compliance setting with VBD
stiffness equal to its reciprocal.

The isolated build and all thirteen CTests passed. Spring tests check known
energy values, both endpoint gradients and diagonal Hessian blocks by finite
differences, force balance, additive assembly, endpoint reversal, trial values,
and degenerate endpoints. Existing implicit-spring, energy-descent, edited cloth,
and 600-frame pendulum checks pass with the new terms.

## Independent solver structs — 1 November 2026

Renamed the types to `XpbdSolver`, `VbdSolver`, and `CudaXpbdSolver`, updating
all callers. Shared particle and distance-constraint declarations now live in
`core/core.hpp`. VBD is independent of `ParticleSolver`: one constraint vector,
no inherited constructor/default overrides, no connectivity cache, and prediction
targets local to `step()`. It scans constraints directly. The exact spring
Hessian, positive-definite block safeguard, and backtracking line search remain.

A fresh build at `/tmp/dynamics-vbd-verify` and all twelve CTests passed. VBD's
unequal-mass spring error was 3.88e-9; its compressed-system objective fell from
36.1841 to 5.47457. The 600-frame elastic-pendulum oracle error was 2.79e-11 m.
Dedicated VBD tests also cover degeneracies, invalid compliance, free fall,
and 180 edited/grabbed frames of a 35-particle cloth. Direct scans favor small
examples; larger systems can restore adjacency later. Native UI checks verified
XPBD/VBD selection, position/velocity continuity, pause, step, reset, and resume.

The normal build was restored after its active watcher overlapped a validation
build. Final numerical validation used the separate build directory.

## Unified XPBD constraints — 1 November 2026

Removed triangles and separate stretch/bend lists from `Xpbd`. It now owns
particles, one constraint list, gravity, and the substep count. CPU XPBD visits
the flat list; CUDA XPBD uses one coloring and one device constraint buffer.
Cloth keeps triangles and material ranges in its scene. Independent test
fixtures retain geometry outside XPBD; specialized CPU reference material
settings remain unchanged. Owned C++/CUDA uses a space before brace initializers.

A clean rebuild and all eleven CTests passed, including the 600-frame CUDA
cloth trajectory, CPU/CUDA timestep parity, reference algorithms, and the
XPBD/VBD pendulum oracle. The clean rebuild replaced a stale VBD object with
missing header dependencies. CUDA ownership checks under compute-sanitizer
reported zero errors and zero leaks. The new unified coloring can change the
order of coupled cloth projections relative to the former two-pass grouping.

## Pendulum XPBD/VBD selection — 1 November 2026

Pendulum now selects CPU XPBD or VBD with the same 1e-6 compliance, 1/60 s
frame timestep, and 30 substeps. VBD uses 10 local iterations. Switching copies
particle position/previous position/velocity/mass without resetting motion.
Reset restores the active solver's particles. The scene remains self-contained.
The existing VBD elastic solver is reused; its numerical algorithm is unchanged.
`Mat3` was removed from owned code in favor of `Eigen::Matrix3d`.

The full build and all eleven CTests passed. New pendulum checks compare both
solvers against the closed-form implicit radial solution over 600 frames:
maximum position error 2.90e-11 m and link extension 1.38e-5 m. The original
hard-XPBD regression still passes. Native mouse/keyboard checks verified actual
XPBD/VBD combo selection, position and velocity continuity, pause, step, reset,
and resume. The VBD screenshot is `/tmp/dynamics-pendulum-vbd.png`.

## Spring controls simplified — 1 November 2026

Removed energy plots/history and time-scale controls at the user's request.
Spring again advances one fixed 1/60 s step per frame, with pause/step/reset.
The 2x ImGui font/style scaling and larger window remain.

## Spring time scale and energy — 1 November 2026

Added an ImGui 0–3x time-scale slider and kinetic/potential/total energy plots.
Playback uses a fixed 1/60 s timestep with 30 substeps; plots retain 600 samples.
The energy helper was checked against 12.5 J initial spring energy, mass-weighted
kinetic energy, and the analytic implicit-oscillator decay: 11.185511 J after
120 steps. The build and existing spring numerical checks passed. Native UI
checks passed zero-speed freeze, paused stepping at zero speed, history reset,
2x playback, and pause/resume. The rolling history was visually checked after
13.6 simulated seconds; it displayed the latest 9.98 seconds. Screenshots are
`/tmp/dynamics-spring-energy-2x.png` and `/tmp/dynamics-spring-energy-history.png`.

## Minimal spring — 1 November 2026

Spring now follows pendulum directly: CPU XPBD, two particles, a compliant
link, zero gravity, explicit drawing/control/input functions, and a fixed camera.
Trails, grabbing, orbit/zoom, compliance/substep controls, and CUDA setup have
been removed from this example. The build and existing XPBD numerical checks
passed. External keyboard and ImGui pause/step/reset/resume checks passed.

## Scene simplification — 1 November 2026

All scenes now use explicit `drawControls`, `handleInput`, scene drawing, and a
plain raylib loop. Simulation state is local to `main`. CLI parsing, smoke and
headless branches, anonymous namespaces, viewer/action structs, and try/catch
have been removed from scene applications. Solver backends are unchanged.
Each rendered frame advances one fixed 1/60 s step, matching pendulum.

The spring, double-pendulum, and cloth CPU/CUDA checks moved to
`tests/xpbd_viewer_tests.cpp`; pendulum checks remain in `tests/pendulum_tests.cpp`.
The full build and all eleven CTests passed. External X11 interaction checks
passed pause/step/reset, particle dragging, and camera orbit for the three XPBD
viewers. Falling-cloth checks also passed hide, solver selection, and dragging;
pan/zoom were exercised. MPM ImGui pause/step/reset/resume checks passed.
Screenshots were inspected under `/tmp/dynamics-simplified-scenes/`.

The sections below record earlier validation, including retired application modes.

## Minimal CPU pendulum — 1 November 2026

`scenes/pendulum.cpp` is a raylib-style example: initialization,
update, draw, cleanup, and `drawPendulum(anchor, bob)` with explicit inputs.
It uses CPU XPBD with simulation/camera state local to `main`, and has no
anonymous namespace, conditional compilation, or try/catch. A small ImGui panel
provides pause, single step, and reset; keyboard shortcuts respect ImGui input capture.

Pendulum now has no command-line parsing, headless branch, smoke mode, or
screenshot capture. CPU physics checks live in `tests/pendulum_tests.cpp`.

Before that simplification, the pendulum headless CTest passed. Over 600 CPU steps the maximum link error
was 3.33e-16 m. A 120-frame viewer run exported its screenshot successfully;
pause, single step, reset, and resume passed native input/image checks.
The binary contains no CUDA XPBD or CUDA allocation symbols.

## Self-contained scenes — 1 November 2026

Each `scenes/*.cpp` owns its setup and complete application. The shared scene
registry, viewer, and check modules have been removed. Scene applications are tested through `--headless`. Numerical solver tests use
ordinary fixtures in `tests/scene_fixtures.cpp`. There is no conditional
compilation or alternate scene build mode. Scene executables link the solver and
rendering libraries directly. There are no headers under `scenes/`.

All eleven CTests and six viewer smoke runs passed after the reorganization.
Pause, step, reset, dragging, orbit, and zoom were checked in all four XPBD
scenes. Headless metrics and rendered scene regions match the earlier version.
Solver source hashes are unchanged.

## Canonical XPBD scenes — 1 November 2026

Added `spring`, `pendulum`, and `double_pendulum` alongside `cloth`, each with a
raylib/ImGui viewer and the CUDA XPBD backend. Each supports display-free headless
checks, JSON metrics, timestep/substep controls, help, and bounded smoke runs.

All eleven CTests passed. Over two seconds, the spring's maximum error against
the continuous oscillator was 0.0255 m (XPBD damping), and the pendulum's maximum
angle error against independent RK4 integration was 0.00435 rad. Over three
seconds, double-pendulum link error stayed below 3.62e-5 m. These checks do not
establish energy conservation or long-trajectory equality for chaotic motion.

Four 120-frame viewer runs passed; screenshots are `/tmp/dynamics-spring.png`,
`dynamics-pendulum.png`, `dynamics-double_pendulum.png`, and `dynamics-cloth-cuda.png`.
Native X11 checks covered pause, single stepping, reset, grabbing, and orbit in
all four scenes; zoom was exercised. The screenshot contact sheet was inspected.
Headless runs passed with `DISPLAY` unset; invalid substeps, fractional step
counts, and nonfinite timesteps were rejected.

## GPU source organization — 1 November 2026

CPU and GPU implementations use separate `.hpp` / `.cpp` pairs: `xpbd`,
`parallel_cloth`, `mpm`, and their `*_gpu` counterparts. CMake compiles only
`*_gpu.cpp` with CUDA; viewers compile as C++ and link one `core` library.
Particle types live with XPBD, CPU reference routines with PBD, and simple error
checks in `utils/cuda_utils.*`. There are no generic core files or CUDA buffer
wrappers. GPU solvers explicitly allocate/free raw device pointers.

All six CTests passed, including ownership checks for move construction,
replacement of live allocations, self moves, moved-from assignment, and cleanup.
CUDA model memcheck with `--leak-check full --quick` reported zero errors and
zero leaked allocations. Three viewer smoke runs and the existing native input
check passed; the input script requires a 1920 × 1080 Xvfb screen.

An earlier same-day source split also passed all three CPU ASan/UBSan suites
with `ASAN_OPTIONS=detect_leaks=0`; those suites were not rerun after this change.

## Raylib/CUDA migration — 30 October 2026

The current viewers use CUDA simulation, raylib rendering, and Eigen/ImGui support.
CPU reference implementations are used by tests; they are not viewer backends.

### Build and numerical checks

Validated on blacknoir with CUDA 12.4, g++ 13.4, Clang 20.1.8, and an RTX 4090.
A fresh `build-default` configured and built with the documented default command:
`CMAKE_CUDA_ARCHITECTURES=native`, `DYNAMICS_REFERENCE_TESTS=OFF`, and
`DYNAMICS_BENCHMARKS=OFF`. At that point, all three viewer translation units compiled as CUDA.

The expanded build enabled CPU reference tests and the projection benchmark.
All six CTest cases passed:

- `xpbd_cuda`: analytic unequal-mass spring; coloring; full-timestep comparison
  against the CPU color-order oracle; pin/grab/material/timestep edits; reset;
  degeneracies; and an independent 600-frame trajectory.
- `models_cuda`: four falling-cloth CUDA schedules against CPU step oracles with
  contacts, pins/grabs, parameter edits, reset, and GPU normal generation;
  direct sphere/ground/friction kernel checks; CUDA MPM timestep comparison,
  P2G mass conservation, material edits, reset, and device invalid-state checks;
  and an independent two-second MPM trajectory with a stiffness edit.
- `cuda_scatter_gather`: analytic/mixed-compliance projection, pin/coloring
  correctness, and CUDA graph/ordinary-launch parity.
- The three optional CPU reference suites for the retained solver algorithms,
  nonlinear Newton/Baraff–Witkin/MPM, and falling-cloth topology/contact/schedules.

In one complete GPU/CPU comparison run, falling cloth's largest step difference
was 5.8e-6 m in position, 5.3e-4 m/s in velocity, and 6.9e-5 in the normal vector.
These are bounded, identical-input timestep comparisons, not a proof of equality
between independently evolved nonlinear trajectories. Atomic addition order can
vary between GPU runs. CUDA MPM's maximum difference across position, velocity,
affine state, and deformation was 2.3e-13 in the tested timesteps. Its independent
2 s trajectory remained finite, non-inverted, and inside the grid walls.

### Memory and viewer checks

`compute-sanitizer --tool memcheck --error-exitcode 1 ./build/cuda_models_tests --quick`
reported zero errors. The quick case covers all falling-cloth schedules, contact,
GPU normals, CUDA MPM transfers/grid updates, reset, and invalid-state handling.
It uses a short timestep workload, rather than the full independent trajectory.

120-frame Xvfb runs passed for CUDA hanging cloth, CUDA MPM, hybrid/Jacobi/gather
falling cloth at 64 × 64 cells, and the serial GPU baseline at 8 × 8 cells.
The falling-cloth native X11 input check verified pause, step, reset, hide/show,
solver switching/reset, dragging, and orbit through image comparisons; pan and
wheel zoom were also exercised. Smoke screenshots are in `/tmp/dynamics-*.png`.

The earlier CPU migration's validation and viewers are preserved under
`../dynamics-archive-2026-10-30/pre-cuda-viewers-2026-10-30/`.
That historical run included a stalled supplemental memory check of the
projection benchmark. It does not provide a verified benchmark memory-check
result; the ordinary numerical/graph tests pass. No new performance comparison,
CUDA/OpenGL interop, or GPU speedup is claimed.

### Reproduce

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

timeout 90s compute-sanitizer --tool memcheck --error-exitcode 1 ./build/cuda_models_tests --quick
xvfb-run -a ./build/cloth --smoke
xvfb-run -a ./build/falling_cloth hybrid --size 64 --smoke
xvfb-run -a ./build/mpm --smoke

# Optional numerical oracles and projection benchmark:
cmake -S . -B build -DDYNAMICS_REFERENCE_TESTS=ON -DDYNAMICS_BENCHMARKS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```
