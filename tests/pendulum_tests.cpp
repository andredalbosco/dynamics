#include "core/xpbd.hpp"
#include "core/vbd.hpp"
#include <cstdio>

int checkHardPendulum() {
    const Eigen::Vector3d anchor {0, 3, 0};
    const Eigen::Vector3d bob = anchor + Eigen::Vector3d(std::sin(0.65), -std::cos(0.65), 0);
    xpbd::Solver pendulum;
    pendulum.particles = {
        {anchor, anchor, Eigen::Vector3d::Zero(), 0},
        {bob, bob, Eigen::Vector3d::Zero(), 1}
    };
    pendulum.constraints = {{0, 1, 1, 0}};

    double maxLinkError = 0;
    for (int frame = 0; frame < 600; ++frame) {
        pendulum.step();
        const auto& p = pendulum.particles[1];
        if (!p.pos.allFinite() || !p.prevPos.allFinite()) return 1;
        maxLinkError = std::max(maxLinkError, std::abs((p.pos - anchor).norm() - 1));
        if ((pendulum.particles[0].pos - anchor).norm() != 0) return 1;
    }
    std::printf("Maximum pendulum link error: %.9g m\n", maxLinkError);
    return maxLinkError < 1e-10 ? 0 : 1;
}

int checkElasticPendulum() {
    const Eigen::Vector3d anchor {0, 3, 0};
    const Eigen::Vector3d bob = anchor + Eigen::Vector3d(std::sin(0.65), -std::cos(0.65), 0);
    xpbd::Solver xpbd;
    xpbd.particles = {
        {anchor, anchor, Eigen::Vector3d::Zero(), 0},
        {bob, bob, Eigen::Vector3d::Zero(), 1}
    };
    constexpr double compliance = 1e-6;
    xpbd.constraints = {{0, 1, 1, compliance}};

    vbd::Solver vbd;
    for (const auto& p : xpbd.particles)
        vbd.particles.push_back({p.pos, p.prevPos, Eigen::Vector3d::Zero(), p.invMass});
    vbd.energies = {{0, 1, 1, 1 / compliance}};
    vbd.substeps = xpbd.substeps;

    // For one free bob, the implicit elastic minimizer has a closed radial solution.
    Eigen::Vector3d position = bob;
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    constexpr double dt = 1.0 / 60;
    const double h = dt / xpbd.substeps;
    const double stiffness = 1 / xpbd.constraints[0].compliance;
    double maxError = 0;
    double maxLinkError = 0;

    for (int frame = 0; frame < 600; ++frame) {
        for (int substep = 0; substep < xpbd.substeps; ++substep) {
            const Eigen::Vector3d target = position + h * velocity + h * h * xpbd.gravity;
            const Eigen::Vector3d direction = target - anchor;
            const double length = direction.norm();
            const double radius = (length + h * h * stiffness) / (1 + h * h * stiffness);
            const Eigen::Vector3d next = anchor + direction * (radius / length);
            velocity = (next - position) / h;
            position = next;
        }

        xpbd.step();
        vbd.step();

        if ((xpbd.particles[0].pos - anchor).norm() != 0 ||
            (vbd.particles[0].pos - anchor).norm() != 0) return 1;
        for (const auto& pos : {xpbd.particles[1].pos, vbd.particles[1].pos}) {
            if (!pos.allFinite()) return 1;
            maxError = std::max(maxError, (pos - position).norm());
            maxLinkError = std::max(maxLinkError, std::abs((pos - anchor).norm() - 1));
        }
    }

    std::printf("Elastic pendulum: implicit solution error %.9g m, link error %.9g m\n",
        maxError, maxLinkError);
    return maxError < 1e-6 && maxLinkError < 3e-5 ? 0 : 1;
}

int checkVelocity() {
    for (int substeps : {30, 7, 1}) {
        for (double dt : {1.0 / 60, 1.0 / 30, 1.0 / 120}) {
            xpbd::Solver solver;
            solver.dt = dt;
            solver.substeps = substeps;
            Eigen::Vector3d position {0, 2, 0};
            Eigen::Vector3d velocity {0.3, -0.2, 0.1};
            solver.particles = {{position, position, velocity, 1}};

            for (int frame = 0; frame < 60; ++frame) {
                solver.dt = frame < 30 ? dt : 2 * dt;
                solver.substeps = frame < 30 ? substeps : substeps + 1;
                const double h = solver.dt / solver.substeps;
                for (int step = 0; step < solver.substeps; ++step) {
                    velocity += h * solver.gravity;
                    position += h * velocity;
                }
                solver.step();
                const auto& p = solver.particles[0];
                if ((p.pos - position).norm() > 1e-9 ||
                    (p.vel - velocity).norm() > 1e-8 ||
                    ((p.pos - p.prevPos) / h - p.vel).norm() > 1e-8) return 1;
            }
            solver.particles[0].invMass = 0;
            solver.particles[0].pos = {1, 3, 0};
            solver.step();
            if (solver.particles[0].pos != solver.particles[0].prevPos ||
                !solver.particles[0].vel.isZero()) return 1;
        }
    }
    return 0;
}

int main() {
    return checkHardPendulum() || checkElasticPendulum() || checkVelocity();
}
