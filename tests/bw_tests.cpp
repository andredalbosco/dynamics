#include "core/bw.hpp"
#include "scene_fixtures.hpp"

#include <cmath>
#include <cstdio>

bool analyticSpring() {
    bw::Solver solver;
    solver.gravity.setZero();
    solver.dt = 0.1;
    solver.substeps = 1;
    solver.particles = {
        {{0, 0, 0}, Eigen::Vector3d::Zero(), {0.2, 0, 0}, 1},
        {{2, 0, 0}, Eigen::Vector3d::Zero(), {-0.7, 0, 0}, 3}
    };
    solver.energies = {{0, 1, 1, 100}};
    solver.step();
    const double length = (1.91 + 4) / 5;
    const double correction = (1.91 - length) / 4;
    return (solver.particles[0].pos - Eigen::Vector3d(0.02 + correction, 0, 0)).norm() < 1e-12 &&
        (solver.particles[1].pos - Eigen::Vector3d(1.93 - 3 * correction, 0, 0)).norm() < 1e-12;
}

bool edgeCases() {
    bw::Solver solver;
    solver.step();
    solver.particles = {
        {{0, 2, 0}, Eigen::Vector3d::Zero(), {0.3, 0.2, -0.1}, 1},
        {{1, 3, 0}, Eigen::Vector3d::Zero(), {1, 2, 3}, 0}
    };
    const Eigen::Vector3d pin = solver.particles[1].pos;
    Eigen::Vector3d position = solver.particles[0].pos;
    Eigen::Vector3d velocity = solver.particles[0].vel;
    for (int frame = 0; frame < 60; ++frame) {
        if (frame == 30) {
            solver.dt = 1.0 / 30;
            solver.substeps = 7;
        }
        const double h = solver.dt / solver.substeps;
        for (int substep = 0; substep < solver.substeps; ++substep) {
            velocity += h * solver.gravity;
            position += h * velocity;
        }
        solver.step();
        if ((solver.particles[0].pos - position).norm() > 1e-10 ||
            (solver.particles[0].vel - velocity).norm() > 1e-10 ||
            solver.particles[1].pos != pin || !solver.particles[1].vel.isZero()) return false;
    }
    // Coincident endpoints and an entirely fixed system.
    solver.particles[0].pos = pin;
    solver.particles[0].vel.setZero();
    solver.gravity.setZero();
    solver.energies = {{0, 1, 1, 100}};
    solver.step();
    if (solver.particles[0].pos != pin) return false;
    solver.particles[0].invMass = 0;
    solver.step();
    return solver.particles[0].pos == pin && solver.particles[0].vel.isZero();
}

bool clothEvolution() {
    const auto fixture = makeClothFixture(5, 7, 0.18, 0.01);
    bw::Solver solver;
    for (const auto& p : fixture.simulation.particles)
        solver.particles.push_back({p.pos, p.prevPos, p.vel, p.invMass});
    for (size_t i = 0; i < fixture.simulation.constraints.size(); ++i) {
        const auto& c = fixture.simulation.constraints[i];
        solver.energies.push_back({c.i, c.j, c.restLength, i < fixture.stretchCount ? 1e6 : 1});
    }
    solver.particles[17].pos.z() += 0.05;
    const Eigen::Vector3d pin = solver.particles[0].pos;
    for (int frame = 0; frame < 90; ++frame) {
        if (frame == 20) solver.particles[17].invMass = 0;
        if (frame == 25) solver.particles[17].invMass = 100;
        if (frame == 50) {
            solver.substeps = 2;
            for (size_t i = 0; i < fixture.stretchCount; ++i) solver.energies[i].stiffness = 1e4;
        }
        const Eigen::Vector3d grab = solver.particles[17].pos;
        solver.step();
        if (solver.particles[0].pos != pin) return false;
        if (frame >= 20 && frame < 25 && solver.particles[17].pos != grab) return false;
        for (const auto& p : solver.particles)
            if (!p.pos.allFinite() || !p.vel.allFinite() || p.pos.norm() > 100) return false;
    }
    return true;
}

int main() {
    if (!analyticSpring() || !edgeCases() || !clothEvolution()) {
        std::fputs("BW checks failed\n", stderr);
        return 1;
    }
    std::puts("BW: unequal-mass spring, free fall, pins, timestep edits, degeneracies, and edited cloth passed");
    return 0;
}
