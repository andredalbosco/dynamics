#pragma once

#include <Eigen/Core>
#include <array>
#include <vector>

// Derived from the NVIDIA (2022) / Ten Minute Physics reference.
// MIT license: licenses/nvidia-cloth.txt.
// Float reference port of the archived Warp/Taichi cloth. Contacts run before
// distance constraints, as in the original; constraint projection can reintroduce
// contact penetration. This model uses hard distance constraints, not XPBD compliance.
struct ParallelCloth {
    using Vec = Eigen::Vector3f;
    enum class Mode { Sequential, Hybrid, Jacobi, Gather };
    struct Particle {
        Vec pos, previous, velocity = Vec::Zero();
        float invMass = 1;
    };
    struct Link { int a, b; float rest; };
    std::vector<Particle> particles;
    std::vector<Vec> restPositions, corrections, normals;
    std::vector<Link> links;
    std::vector<Eigen::Vector3i> triangles;
    std::vector<std::vector<int>> incident;
    std::array<int, 6> passOffsets {};
    int numX, numY;
    int substeps = 30;
    float jacobiScale = 0.2f;
    Vec gravity {0, -10, 0};
    Vec sphereCenter {0, 1.5f, 0};
    float sphereRadius = 0.5f;
    float thickness = 0.001f, friction = 0.01f;
    bool sphereEnabled = true, groundEnabled = true;
    Mode mode = Mode::Hybrid;

    explicit ParallelCloth(int cellsX = 140, int cellsY = 140,
        float spacing = 0.01f, float height = 2.2f);
    void reset();
    void integrate(float h);
    void solveRange(int first, int last, bool jacobi);
    void solveGather();
    void step(float dt);
    void updateNormals();
    float maxRelativeError() const;
};
