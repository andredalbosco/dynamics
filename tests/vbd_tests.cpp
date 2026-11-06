#include "core/vbd.hpp"
#include "scene_fixtures.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

vbd::Solver makePair() {
    vbd::Solver solver;
    solver.particles = {
        {{0, 0, 0}, {0, 0, 0}, Eigen::Vector3d::Zero(), 1},
        {{2, 0, 0}, {2, 0, 0}, Eigen::Vector3d::Zero(), 3}
    };
    solver.energies = {{0, 1, 1, 100}};
    solver.gravity.setZero();
    solver.substeps = 1;
    solver.dt = 0.1;
    return solver;
}

double objective(const vbd::Solver& solver, const std::vector<Eigen::Vector3d>& targets, double h) {
    double energy = 0;
    for (size_t i = 0; i < solver.particles.size(); ++i) {
        const auto& p = solver.particles[i];
        if (p.invMass > 0)
            energy += (p.pos - targets[i]).squaredNorm() / (2 * p.invMass * h * h);
    }
    for (const auto& term : solver.energies) {
        const double extension = (solver.particles[term.i].pos - solver.particles[term.j].pos).norm() - term.restLength;
        energy += 0.5 * term.stiffness * extension * extension;
    }
    return energy;
}

bool analyticSpring() {
    auto solver = makePair();
    solver.particles[0].vel = {0.2, 0, 0};
    solver.particles[1].vel = {-0.7, 0, 0};
    constexpr double h = 0.1;
    // Exact coordinate updates for one sweep: move a, then b with the new a.
    const double a = (0.02 / (h * h) + 100 * (1.93 - 1)) / (1 / (h * h) + 100);
    const double b = (1.93 / (3 * h * h) + 100 * (a + 1)) / (1 / (3 * h * h) + 100);

    solver.step();
    const double error = std::max(
        (solver.particles[0].pos - Eigen::Vector3d(a, 0, 0)).norm(),
        (solver.particles[1].pos - Eigen::Vector3d(b, 0, 0)).norm());
    std::printf("VBD unequal-mass single-sweep spring error: %.9g\n", error);
    return error < 1e-8;
}

bool descent() {
    auto initial = makePair();
    initial.particles[1].pos = {0.4, 0.2, -0.3};
    initial.particles.push_back({{1.7, -0.3, 0.7}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 2});
    initial.energies.push_back({1, 2, 1, 100});

    std::vector<Eigen::Vector3d> targets;
    for (const auto& p : initial.particles) targets.push_back(p.pos);
    const double start = objective(initial, targets, initial.dt);
    initial.step();
    const double current = objective(initial, targets, initial.dt);
    std::printf("VBD single-sweep compressed-system objective: %.9g -> %.9g\n", start, current);
    return std::isfinite(current) && current < start;
}

bool edgeCases() {
    auto falling = makePair();
    falling.energies.clear();
    falling.gravity = {0, -9.81, 0};
    falling.step();
    if (std::abs(falling.particles[0].pos.y() + 0.0981) > 1e-12 ||
        std::abs(falling.particles[0].vel.y() + 0.981) > 1e-12) return false;

    auto solver = makePair();
    solver.particles[1].pos.setZero();
    solver.step();
    for (const auto& p : solver.particles)
        if (!p.pos.allFinite() || !p.vel.allFinite()) return false;
    for (auto& p : solver.particles) p.invMass = 0;
    const auto particles = solver.particles;
    solver.step();
    for (size_t i = 0; i < particles.size(); ++i)
        if (solver.particles[i].pos != particles[i].pos) return false;

    vbd::Solver empty;
    empty.step();
    return true;
}

bool clothEvolution() {
    auto fixture = makeClothFixture(5, 7, 0.18, 0.01);
    vbd::Solver solver;
    for (const auto& p : fixture.simulation.particles)
        solver.particles.push_back({p.pos, p.prevPos, Eigen::Vector3d::Zero(), p.invMass});
    for (size_t i = 0; i < fixture.simulation.constraints.size(); ++i) {
        const auto& c = fixture.simulation.constraints[i];
        const double stiffness = i < fixture.stretchCount ? 1e6 : 1 / c.compliance;
        solver.energies.push_back({c.i, c.j, c.restLength, stiffness});
    }
    solver.substeps = 4;

    const Eigen::Vector3d left = solver.particles[0].pos;
    const Eigen::Vector3d right = solver.particles[6].pos;
    solver.particles[17].pos.z() = 0.15;

    for (int frame = 0; frame < 180; ++frame) {
        if (frame == 60) {
            solver.particles[17].pos += Eigen::Vector3d(0.1, 0.1, 0.2);
            solver.particles[17].invMass = 0;
        }
        if (frame == 70) solver.particles[17].invMass = 100;
        if (frame == 100) {
            solver.substeps = 2;
            for (size_t i = 0; i < fixture.stretchCount; ++i) solver.energies[i].stiffness = 1e4;
        }
        const Eigen::Vector3d grab = solver.particles[17].pos;
        solver.step();
        if (frame >= 60 && frame < 70 && solver.particles[17].pos != grab) return false;
        if (solver.particles[0].pos != left || solver.particles[6].pos != right) return false;
        for (const auto& p : solver.particles)
            if (!p.pos.allFinite() || !p.vel.allFinite() || p.pos.norm() >= 100 || p.vel.norm() >= 1e4)
                return false;
    }
    std::puts("VBD 180-frame cloth: finite, fixed pins, grabs and material edits passed");
    return true;
}

int main() {
    if (!analyticSpring() || !descent() || !edgeCases() || !clothEvolution()) {
        std::fputs("VBD checks failed\n", stderr);
        return 1;
    }
    return 0;
}
