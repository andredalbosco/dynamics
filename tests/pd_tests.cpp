#include "core/pd.hpp"
#include "scene_fixtures.hpp"

#include <cmath>
#include <cstdio>

void makePair(pd::Solver& solver) {
    solver.particles.resize(2);
    solver.particles.pos << 0, 0, 0, 2, 0, 0;
    solver.particles.prevPos = solver.particles.pos;
    solver.particles.invMass << 1, 3;
    solver.energies = {{0, 1, 1, 100}};
    solver.gravity.setZero();
    solver.substeps = 1;
    solver.dt = 0.1;
    solver.factorize();
}

double energy(const pd::Solver& solver, double h) {
    double value = 0;
    auto& p = solver.particles;
    for (int i = 0; i < p.pos.rows(); ++i) {
        if (p.invMass[i] > 0)
            value += (p.pos.row(i) - p.inertialPos.row(i)).squaredNorm() / (2 * p.invMass[i] * h * h);
    }
    for (const auto& term : solver.energies) {
        const double extension = (p.pos.row(term.i) - p.pos.row(term.j)).norm() - term.restLength;
        value += 0.5 * term.stiffness * extension * extension;
    }
    return value;
}

bool analyticSpring() {
    pd::Solver solver;
    makePair(solver);
    solver.particles.vel.row(0) << 0.2, 0, 0;
    solver.particles.vel.row(1) << -0.7, 0, 0;
    const double h = solver.dt;
    const double length = (1.91 + h * h * 100 * 4) / (1 + h * h * 100 * 4);
    const double correction = (1.91 - length) / 4;

    solver.step();
    const double error = std::max(
        (solver.particles.pos.row(0) - Eigen::RowVector3d(0.02 + correction, 0, 0)).norm(),
        (solver.particles.pos.row(1) - Eigen::RowVector3d(1.93 - 3 * correction, 0, 0)).norm());
    std::printf("PD unequal-mass implicit spring error: %.9g\n", error);
    return error < 1e-12;
}

bool gradientChecks() {
    pd::Solver solver;
    makePair(solver);
    auto& p = solver.particles;
    p.resize(3);
    p.pos << 0, 0, 0, 2, 0, 0, 0.5, 0.8, -0.2;
    p.invMass << 0, 3, 2;
    p.prevPos = p.pos;
    p.inertialPos = p.pos;
    solver.energies.push_back({1, 2, 1, 50});
    p.pos.row(1) += Eigen::RowVector3d(0.2, -0.1, 0.3);
    solver.factorize();
    Eigen::MatrixXd gradient;
    solver.objectiveGradient(gradient);

    for (int i = 0; i < p.pos.rows(); ++i) {
        if (p.invMass[i] == 0) {
            if (!gradient.row(i).isZero()) return false;
            continue;
        }
        for (int axis = 0; axis < 3; ++axis) {
            const double position = p.pos(i, axis);
            p.pos(i, axis) = position + 1e-6;
            const double plus = energy(solver, solver.dt);
            p.pos(i, axis) = position - 1e-6;
            const double minus = energy(solver, solver.dt);
            p.pos(i, axis) = position;
            if (std::abs((plus - minus) / 2e-6 - gradient(i, axis)) > 1e-5) return false;
        }
    }
    return true;
}

bool descent() {
    pd::Solver solver;
    makePair(solver);
    auto& p = solver.particles;
    p.resize(3);
    p.pos << 0, 0, 0, 0.4, 0.2, -0.3, 1.7, -0.3, 0.7;
    p.invMass << 1, 3, 2;
    p.prevPos = p.pos;
    p.inertialPos = p.pos;
    solver.energies.push_back({1, 2, 1, 100});
    solver.factorize();
    double previous = energy(solver, solver.dt);
    const double start = previous;
    for (int step = 0; step < 60; ++step) {
        Eigen::MatrixXd gradient;
        solver.objectiveGradient(gradient);
        solver.particles.pos += solver.cache.factor.solve(-gradient);
        const double current = energy(solver, solver.dt);
        if (!std::isfinite(current) || current > previous + 1e-10) return false;
        previous = current;
    }
    std::printf("PD compressed-system objective: %.9g -> %.9g\n", start, previous);
    return previous < 0.9 * start;
}

bool editsAndPins() {
    pd::Solver solver;
    makePair(solver);
    solver.particles.invMass[0] = 0;
    for (int frame = 0; frame < 10; ++frame) {
        // Matrix inputs and fixed positions can change between steps.
        solver.particles.pos.row(0) << 0.03 * frame, 0.01 * frame, 0;
        solver.particles.invMass[1] = frame % 2 ? 2 : 3;
        solver.energies[0].stiffness = frame % 2 ? 50 : 100;
        solver.energies[0].restLength = frame % 2 ? 1.2 : 1;
        solver.dt = frame % 2 ? 0.05 : 0.1;
        const double h = solver.dt;
        const Eigen::RowVector3d pin = solver.particles.pos.row(0);
        const auto& p = solver.particles;
        const Eigen::RowVector3d target = p.pos.row(1) + h * p.vel.row(1);
        const Eigen::RowVector3d direction = target - pin;
        const double length = direction.norm();
        const double scale = h * h * p.invMass[1] * solver.energies[0].stiffness;
        const double radius = (length + scale * solver.energies[0].restLength) / (1 + scale);
        const Eigen::RowVector3d expected = pin + direction * (radius / length);
        solver.factorize();
        solver.step();
        if (solver.particles.pos.row(0) != pin || (solver.particles.pos.row(1) - expected).norm() > 1e-12)
            return false;
    }
    return true;
}

bool edgeCases() {
    const Eigen::RowVector3d origin = Eigen::RowVector3d::Zero();
    const Eigen::RowVector3d previousDelta {0, 2, 0};
    Eigen::RowVector3d gradient = Eigen::RowVector3d::Zero();
    pd::springEnergyGradient(origin, 1, 100, previousDelta, gradient);
    if ((gradient - Eigen::RowVector3d(0, -100, 0)).norm() > 1e-12) return false;
    gradient.setZero();
    pd::springEnergyGradient(origin, 1, 100, origin, gradient);
    if ((gradient - Eigen::RowVector3d(-100, 0, 0)).norm() > 1e-12) return false;

    pd::Solver solver;
    makePair(solver);
    solver.energies.clear();
    solver.gravity = {0, -9.81, 0};
    solver.substeps = 10;
    solver.factorize();
    solver.step();
    auto& p = solver.particles;
    if (std::abs(p.pos(0, 1) + 0.053955) > 1e-12 || std::abs(p.vel(0, 1) + 0.981) > 1e-12)
        return false;

    makePair(solver);
    p.pos.row(1) = p.pos.row(0);
    solver.step();
    if (!p.pos.allFinite() || !p.vel.allFinite()) return false;
    p.invMass.setZero();
    const Eigen::MatrixXd positions = p.pos;
    solver.factorize();
    solver.step();
    if (p.pos != positions || !p.vel.isZero()) return false;
    p.resize(0);
    solver.energies.clear();
    solver.factorize();
    solver.step();
    return true;
}

bool clothEvolution() {
    const auto fixture = makeClothFixture(5, 7, 0.18, 0.01);
    pd::Solver solver;
    auto& p = solver.particles;
    p.resize(int(fixture.simulation.particles.size()));
    for (int i = 0; i < p.pos.rows(); ++i) {
        p.pos.row(i) = fixture.simulation.particles[i].pos.transpose();
        p.invMass[i] = fixture.simulation.particles[i].invMass;
    }
    p.prevPos = p.pos;
    for (size_t i = 0; i < fixture.simulation.constraints.size(); ++i) {
        const auto& c = fixture.simulation.constraints[i];
        solver.energies.push_back({c.i, c.j, c.restLength, i < fixture.stretchCount ? 1e6 : 1 / c.compliance});
    }
    solver.factorize();
    const Eigen::RowVector3d left = p.pos.row(0);
    const Eigen::RowVector3d right = p.pos.row(6);
    for (int frame = 0; frame < 180; ++frame) {
        if (frame == 60) {
            p.pos.row(17) += Eigen::RowVector3d(0.1, 0.1, 0.2);
            p.invMass[17] = 0;
        }
        if (frame == 70) p.invMass[17] = 100;
        if (frame == 100) {
            solver.substeps = 2;
            for (size_t i = 0; i < fixture.stretchCount; ++i) solver.energies[i].stiffness = 1e4;
        }
        if (frame == 60 || frame == 70 || frame == 100)
            solver.factorize();
        const Eigen::RowVector3d grab = p.pos.row(17);
        solver.step();
        if (p.pos.row(0) != left || p.pos.row(6) != right) return false;
        if (frame >= 60 && frame < 70 && p.pos.row(17) != grab) return false;
        if (!p.pos.allFinite() || !p.vel.allFinite() ||
            p.pos.rowwise().norm().maxCoeff() > 100 || p.vel.rowwise().norm().maxCoeff() > 1e4)
            return false;
    }
    return true;
}

int main() {
    if (!analyticSpring() || !gradientChecks() || !descent() || !editsAndPins() ||
        !edgeCases() || !clothEvolution()) {
        std::fputs("PD checks failed\n", stderr);
        return 1;
    }
    std::puts("PD: gradient, descent, edits, pins, degeneracies, and 180-frame cloth passed");
    return 0;
}
