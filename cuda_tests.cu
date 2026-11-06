#include "core/xpbd_gpu.hpp"

#include "tests/scene_fixtures.hpp"
#include <cstdio>

static void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

static void checkColors(const xpbd::ConstraintColors& colors,
    const std::vector<xpbd::DistanceConstraint>& constraints, size_t count) {
    std::vector<bool> seen(constraints.size(), false);
    for (size_t color = 1; color < colors.offsets.size(); ++color) {
        std::vector<bool> used(count, false);
        for (int j = colors.offsets[color - 1]; j < colors.offsets[color]; ++j) {
            const int index = colors.order[j];
            const auto& c = constraints[index];
            require(!seen[index] && !used[c.i] && !used[c.j], "Invalid constraint coloring");
            seen[index] = used[c.i] = used[c.j] = true;
        }
    }
    for (bool value : seen)
        require(value, "Missing constraint in coloring");
}

static xpbd::Solver coloredReference(const xpbd::Solver& original, const xpbd::ConstraintColors& colors) {
    xpbd::Solver result = original;
    const double h = original.dt / original.substeps;
    for (auto& p : result.particles)
        p.vel = (p.pos - p.prevPos) / h;
    for (size_t i = 0; i < colors.order.size(); ++i)
        result.constraints[i] = original.constraints[colors.order[i]];
    return result;
}

static double compare(const xpbd::CudaSolver& gpu, const xpbd::Solver& cpu) {
    double error = 0;
    for (size_t i = 0; i < cpu.particles.size(); ++i) {
        require(gpu.particles[i].pos.allFinite() && gpu.particles[i].prevPos.allFinite(),
            "Nonfinite GPU state");
        error = std::max(error, (gpu.particles[i].pos - cpu.particles[i].pos).norm());
        const Eigen::Vector3d gpuVelocity = (gpu.particles[i].pos - gpu.particles[i].prevPos) / gpu.previousH;
        const Eigen::Vector3d cpuVelocity = (cpu.particles[i].pos - cpu.particles[i].prevPos) / (cpu.dt / cpu.substeps);
        error = std::max(error, (gpuVelocity - cpuVelocity).norm());
        error = std::max(error, (gpu.particles[i].prevPos - cpu.particles[i].prevPos).norm());
        require(gpu.particles[i].invMass == cpu.particles[i].invMass, "Inverse mass changed");
    }
    auto compareLambda = [&](const auto& actual, const auto& reference, const auto& colors) {
        for (size_t i = 0; i < colors.order.size(); ++i)
            error = std::max(error, std::abs(actual[colors.order[i]].lambda - reference[i].lambda));
    };
    compareLambda(gpu.constraints, cpu.constraints, gpu.colors);
    require(std::isfinite(error) && error < 1e-7, "CPU colored / CUDA timestep mismatch");
    return error;
}

static int tests() {
    // An isolated unequal-mass compliant spring has an analytic one-step solution.
    auto spring = makeCloth<xpbd::Solver>(2, 2, 1, 1);
    spring.particles.resize(2);
    spring.particles[0] = {Eigen::Vector3d {0, 0, 0}, Eigen::Vector3d {-0.0025, 0, 0}, Eigen::Vector3d::Zero(), 1};
    spring.particles[1] = {Eigen::Vector3d {2, 0, 0}, Eigen::Vector3d {2.001, 0, 0}, Eigen::Vector3d::Zero(), 0.5};
    spring.constraints = {{0, 1, 1, 1e-4}};
    spring.gravity.setZero();
    spring.substeps = 1;
    xpbd::CudaSolver analytic {spring};
    constexpr double h = 0.01;
    const double dlambda = -(2 - 0.35 * h - 1) / (1.5 + 1e-4 / (h * h));
    analytic.step(h);
    require(std::abs(analytic.particles[0].pos.x() - (0.25 * h - dlambda)) < 1e-12,
        "Analytic unequal-mass spring (a)");
    require(std::abs(analytic.particles[1].pos.x() - (2 - 0.1 * h + 0.5 * dlambda)) < 1e-12,
        "Analytic unequal-mass spring (b)");

    double maximumError = 0;
    for (double compliance : {0.0, 1e-6, 1e-4}) {
        auto gpu = makeCloth<xpbd::CudaSolver>(17, 21, 0.18, 0.01);
        const size_t stretchCount = makeClothFixture(17, 21, 0.18, 0.01).stretchCount;
        for (size_t i = 0; i < stretchCount; ++i)
            gpu.constraints[i].compliance = compliance;
        gpu.particles.back().prevPos = gpu.particles.back().pos - Eigen::Vector3d {0.1, 0.2, 0.3} * (1.0 / 60 / gpu.substeps);
        checkColors(gpu.colors, gpu.constraints, gpu.particles.size());
        auto cpu = coloredReference(gpu, gpu.colors);
        const Eigen::Vector3d pin = gpu.particles.front().pos;
        for (int frame = 0; frame < 120; ++frame) {
            // Compare each complete GPU timestep from the same CPU input state.
            // Long nonlinear trajectories can amplify tiny floating-point differences.
            gpu.particles = cpu.particles;
            if (frame == 20) { // Exercise viewer edits: grab and move a free particle.
                gpu.particles.back().invMass = cpu.particles.back().invMass = 0;
                gpu.particles.back().pos += Eigen::Vector3d {0.1, 0.05, 0.1};
                cpu.particles.back().pos = gpu.particles.back().pos;
                gpu.particles.back().prevPos = gpu.particles.back().pos;
                cpu.particles.back().prevPos = cpu.particles.back().pos;
            }
            if (frame == 25)
                gpu.particles.back().invMass = cpu.particles.back().invMass = 100;
            if (frame == 40) {
                gpu.substeps = cpu.substeps = 7;
                for (size_t i = 0; i < gpu.colors.order.size(); ++i) {
                    const int index = gpu.colors.order[i];
                    if (size_t(index) >= stretchCount) {
                        gpu.constraints[index].compliance = 0.05;
                        cpu.constraints[i].compliance = 0.05;
                    }
                }
            }
            gpu.step(1.0 / 60);
            cpu.step();
            try {
                maximumError = std::max(maximumError, compare(gpu, cpu));
            } catch (...) {
                std::fprintf(stderr, "compliance %.3g frame %d\n", compliance, frame);
                throw;
            }
            require((gpu.particles.front().pos - pin).norm() == 0, "Pinned particle moved");
        }
        // Reset uses move assignment just as the viewer does; old allocations must be released.
        gpu = makeCloth<xpbd::CudaSolver>(17, 21, 0.18, 0.01);
        cpu = coloredReference(gpu, gpu.colors);
        gpu.step(1.0 / 60);
        cpu.step();
        maximumError = std::max(maximumError, compare(gpu, cpu));
    }
    // Independent GPU evolution, without resetting from the CPU reference.
    auto evolution = makeCloth<xpbd::CudaSolver>(17, 21, 0.18, 0.01);
    const Eigen::Vector3d leftPin = evolution.particles.front().pos;
    const Eigen::Vector3d rightPin = evolution.particles[20].pos;
    double maximumStretch = 0;
    const size_t stretchCount = makeClothFixture(17, 21, 0.18, 0.01).stretchCount;
    for (int frame = 0; frame < 600; ++frame) {
        evolution.step(1.0 / 60);
        for (const auto& p : evolution.particles)
            require(p.pos.allFinite() && p.prevPos.allFinite(), "Nonfinite long GPU trajectory");
        require((evolution.particles.front().pos - leftPin).norm() == 0, "Left pin drift");
        require((evolution.particles[20].pos - rightPin).norm() == 0,
            "Right pin drift");
        for (size_t i = 0; i < stretchCount; ++i) {
            const auto& c = evolution.constraints[i];
            maximumStretch = std::max(maximumStretch,
                std::abs((evolution.particles[c.i].pos - evolution.particles[c.j].pos).norm()
                / c.restLength - 1));
        }
    }
    require(maximumStretch < 0.25, "Excessive hard-cloth stretch");
    std::printf("Independent 600-frame GPU trajectory: finite, fixed pins, "
        "peak relative stretch %.3g\n", maximumStretch);
    // Coincident endpoints and two pinned particles must not divide by zero.
    for (bool coincident : {false, true}) {
        auto fixture = spring;
        if (coincident) {
            fixture.particles[1].pos = fixture.particles[0].pos;
        } else {
            for (auto& p : fixture.particles)
                p.invMass = 0;
        }
        for (auto& p : fixture.particles)
            p.prevPos = p.pos;
        fixture.dt = h;
        xpbd::CudaSolver gpu {fixture};
        gpu.step(h);
        fixture.step();
        compare(gpu, fixture);
    }
    std::printf("CUDA tests passed: analytic spring, coloring, timestep parity, edits, pins, "
        "reset, degeneracies; max timestep error %.3g\n", maximumError);
    return 0;
}

int main() {
    try {
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        require(devices > 0, "No CUDA device available");
        return tests();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA XPBD: %s\n", error.what());
        return 1;
    }
}
