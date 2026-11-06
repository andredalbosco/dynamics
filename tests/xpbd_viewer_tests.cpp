#include "scene_fixtures.hpp"
#include "core/xpbd_gpu.hpp"
#include <cstdio>
#include <cstring>

struct Metrics {
    double cpuError = 0;
    double pinError = 0;
    double hardLinkError = 0;
    double distanceError = 0;
    double motion = 0;
};

xpbd::Solver referenceStep(const xpbd::CudaSolver& simulation, double dt) {
    xpbd::Solver reference = simulation;
    reference.dt = dt;
    const double h = simulation.previousH > 0 ? simulation.previousH : dt / simulation.substeps;
    for (auto& p : reference.particles)
        p.vel = (p.pos - p.prevPos) / h;
    const auto reorder = [](auto& destination, const auto& source, const auto& colors) {
        for (size_t i = 0; i < source.size(); ++i) {
            destination[i] = source[colors.order[i]];
        }
    };
    reorder(reference.constraints, simulation.constraints, simulation.colors);
    reference.step();
    return reference;
}

void measureParticles(Metrics& metrics, const xpbd::Solver& current, const xpbd::Solver& reference, const xpbd::Solver& initial) {
    for (size_t i = 0; i < current.particles.size(); ++i) {
        const auto& p = current.particles[i];
        if (!p.pos.allFinite() || !p.prevPos.allFinite()) {
            throw std::runtime_error("Nonfinite XPBD state");
        }
        metrics.cpuError = std::max({metrics.cpuError,
            (p.pos - reference.particles[i].pos).norm(),
            (p.prevPos - reference.particles[i].prevPos).norm()});
        const double displacement = (p.pos - initial.particles[i].pos).norm();
        if (p.invMass == 0) {
            metrics.pinError = std::max(metrics.pinError, displacement);
        } else {
            metrics.motion = std::max(metrics.motion, displacement);
        }
    }
}

void measureLinks(Metrics& metrics, const xpbd::Solver& state, const std::vector<xpbd::DistanceConstraint>& constraints) {
    for (const auto& c : constraints) {
        const double relative = std::abs((state.particles[c.i].pos -
            state.particles[c.j].pos).norm() / c.restLength - 1);
        metrics.distanceError = std::max(metrics.distanceError, relative);
        if (c.compliance == 0) {
            metrics.hardLinkError = std::max(metrics.hardLinkError, relative);
        }
    }
}


int checkScene(const char* name, const xpbd::Solver& initial, int steps, double dt) {
    xpbd::CudaSolver simulation {initial};
    Metrics metrics;
    for (int frame = 0; frame < steps; ++frame) {
        const xpbd::Solver reference = referenceStep(simulation, dt);
        simulation.step(dt);
        measureParticles(metrics, simulation, reference, initial);
        measureLinks(metrics, simulation, simulation.constraints);
    }
    std::printf("{\"scene\":\"%s\",\"steps\":%d,\"dt\":%.17g,\"substeps\":%d,"
        "\"time\":%.17g,\"cpu_step_error\":%.9g,\"pin_error\":%.9g,"
        "\"hard_link_relative_error\":%.9g,\"distance_relative_error\":%.9g,\"motion\":%.9g}\n",
        name, steps, dt, initial.substeps, steps * dt, metrics.cpuError, metrics.pinError,
        metrics.hardLinkError, metrics.distanceError, metrics.motion);
    if (metrics.cpuError > 1e-7 || metrics.pinError != 0 ||
        metrics.hardLinkError > 0.25 || !(metrics.motion > 0)) {
        throw std::runtime_error("XPBD scene check failed");
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    if (std::strcmp(argv[1], "spring") == 0)
        return checkScene("spring", makeSpring(), 120, 1.0 / 60);
    if (std::strcmp(argv[1], "double_pendulum") == 0)
        return checkScene("double_pendulum", makeDoublePendulum(), 120, 1.0 / 60);
    if (std::strcmp(argv[1], "cloth") == 0)
        return checkScene("cloth", makeCloth(17, 21, 0.18, 0.01), 120, 1.0 / 60);
    return 2;
}
