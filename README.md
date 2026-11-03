# dynamics

Minimal C++17 implementations of XPBD, VBD, AVBD, and Projective Dynamics (PD),
with Eigen math, raylib rendering, and ImGui controls.

## Build and run

Requires CMake 3.24+, a C++17 compiler, and raylib desktop dependencies. On Ubuntu:

```sh
sudo apt install build-essential cmake git libasound2-dev libx11-dev libxrandr-dev libxi-dev libgl1-mesa-dev libglu1-mesa-dev libxcursor-dev libxinerama-dev
git submodule update --init --depth 1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/spring
./build/pendulum
```

Dependencies are pinned as Git submodules under `external/`: Eigen, raylib,
Dear ImGui, and rlImGui. No CUDA toolkit is required.

## Examples

- **Spring:** one fixed anchor and a 1 kg bob, 1 m rest length, stiffness
  100 N/m (compliance 0.01), and no gravity. Select XPBD, VBD, PD, or AVBD.
- **Pendulum:** one fixed anchor and a 1 kg bob, 1 m rest length, gravity
  9.81 m/s², and initial angle 0.65 rad. Select XPBD or AVBD with a hard
  distance constraint, or VBD with an elastic link of stiffness 1e6 N/m.

Each example advances its solvers together at a timestep of 1/60 s with 10
substeps; the selector controls which result is drawn. Use the panel or Space
for pause, N to step, and R to reset. Both scenes use a fixed camera.
AVBD hard constraints can retain residual error with one sweep per substep.
There is no collision handling or energy-conservation correction.

## Source and checks

`core/{xpbd,vbd,avbd,pd}.{hpp,cpp}` contain the four solvers. Each scene in
`scenes/` owns its setup, controls, stepping, drawing, and main loop. PD requires
`factorize()` after changing timestep, masses, pins, stiffness, or topology.

```sh
ctest --test-dir build --output-on-failure
```

Tests compare all four spring solvers with a closed-form implicit-Euler
oscillator, check XPBD/VBD pendulum motion against an implicit radial solution,
verify hard XPBD link lengths and pins, and check VBD spring derivatives by
finite differences. These checks do not establish energy conservation.
