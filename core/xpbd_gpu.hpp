#pragma once

#include "xpbd.hpp"
#include "utils/cuda_utils.hpp"

namespace xpbd {

struct DeviceParticle { double x[3], previous[3], invMass; };

struct ConstraintColors {
    std::vector<int> order;
    std::vector<int> offsets {0};

    ConstraintColors(const std::vector<DistanceConstraint>& constraints, size_t particleCount);
};

struct CudaSolver : Solver {
    double previousH = 0;
    ConstraintColors colors;
    DeviceParticle* deviceParticles = nullptr;
    DistanceConstraint* deviceConstraints = nullptr;

    std::vector<DeviceParticle> hostParticles;
    std::vector<DistanceConstraint> hostConstraints;

    explicit CudaSolver(Solver state);
    ~CudaSolver();
    CudaSolver(const CudaSolver&) = delete;
    CudaSolver& operator=(const CudaSolver&) = delete;
    CudaSolver(CudaSolver&& other) noexcept;
    CudaSolver& operator=(CudaSolver&& other) noexcept;

    void step(double dt);
};

} // namespace xpbd
