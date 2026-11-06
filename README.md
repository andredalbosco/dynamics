# dynamics

- XPBD — Extended Position-Based Dynamics
- VBD — Vertex Block Descent
- AVBD — Augmented Vertex Block Descent
- PD — Projective Dynamics

![Spring](docs/images/spring.png)

![Pendulum](docs/images/pendulum.png)

## Build and examples

```sh
git submodule update --init
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
ctest --test-dir build --output-on-failure
./build/spring
./build/cloth
./build/falling_cloth
./build/mpm
```

The current build requires C++17, CUDA, Eigen, raylib and ImGui. CPU examples
compare XPBD, VBD, AVBD, Projective Dynamics and Baraff–Witkin. Additional
examples include CUDA XPBD, parallel cloth and material point simulation.
See [validation](VALIDATION.md), [references](references.md), and the
[scatter/gather benchmark](benchmarks/xpbd_scatter_gather/README.md).
Source provenance is recorded in [docs/source-provenance.json](docs/source-provenance.json);
the adapted cloth code retains its [NVIDIA license](licenses/nvidia-cloth.txt).
