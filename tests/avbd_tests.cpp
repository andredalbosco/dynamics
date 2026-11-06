#include "core/avbd.hpp"
#include "scene_fixtures.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <cstdio>

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

avbd::Solver makePair(double compliance = 0.01) {
    avbd::Solver solver;
    solver.particles = {
        {{0, 0, 0}, {0, 0, 0}, Eigen::Vector3d::Zero(), 1},
        {{2, 0, 0}, {2, 0, 0}, Eigen::Vector3d::Zero(), 3}
    };
    solver.constraints = {{0, 1, 1, compliance}};
    solver.gravity.setZero();
    solver.substeps = 1;
    solver.dt = 0.1;
    return solver;
}

void analyticSweep(double compliance, double length, double velocityA, double velocityB) {
    auto solver = makePair(compliance);
    solver.particles[1].pos.x() = length;
    solver.particles[0].vel = {velocityA, 0, 0};
    solver.particles[1].vel = {velocityB, 0, 0};
    constexpr double h = 0.1;
    const double targetA = h * velocityA;
    const double targetB = length + h * velocityB;
    const double restLength = compliance == 0 ? 1 + solver.alpha * (length - 1) : 1;
    const double penalty = compliance == 0 ? 1000 : 1 / compliance;

    // Closed-form coordinate minima for a 1D spring: first a, then b.
    const double a = (targetA / (h * h) + penalty * (targetB - restLength))
        / (1 / (h * h) + penalty);
    const double b = (targetB / (3 * h * h) + penalty * (a + restLength))
        / (1 / (3 * h * h) + penalty);

    solver.step();
    const double error = std::max(
        (solver.particles[0].pos - Eigen::Vector3d(a, 0, 0)).norm(),
        (solver.particles[1].pos - Eigen::Vector3d(b, 0, 0)).norm()
    );
    std::printf("AVBD single-sweep spring error: %.9g\n", error);
    check(error < 1e-8, "single-sweep spring with incoming velocity");
}

void constraintState() {
    auto solver = makePair(1e-6);
    solver.step();
    check(
        solver.constraints[0].penalty > 1000 && solver.constraints[0].penalty <= 1e6 && solver.constraints[0].lambda == 0,
        "finite AVBD ramps penalty without a multiplier"
    );

    auto hard = makePair(0);
    hard.step();
    const double residual = (hard.particles[1].pos - hard.particles[0].pos).norm() - 1.95;
    check(
        std::abs(hard.constraints[0].lambda - 1000 * residual) < 1e-10 &&
        std::abs(hard.constraints[0].penalty - (1000 + hard.beta * std::abs(residual))) < 1e-10,
        "hard AVBD updates lambda and penalty from the relaxed residual"
    );

    const double lambda = hard.constraints[0].lambda;
    const double penalty = hard.constraints[0].penalty;
    for (auto& c : hard.constraints) c.warmStart(hard.particles, hard.alpha, hard.gamma);
    check(
        std::abs(hard.constraints[0].lambda - hard.alpha * hard.gamma * lambda) < 1e-12 &&
        std::abs(hard.constraints[0].penalty - hard.gamma * penalty) < 1e-9,
        "AVBD warm start decay"
    );

    hard.constraints[0].compliance = 0.01;
    hard.step();
    check(
        hard.constraints[0].lambda == 0 && hard.constraints[0].penalty <= 100,
        "hard-to-finite material edit clears lambda"
    );
}

void edgeCases() {
    auto falling = makePair();
    falling.constraints.clear();
    falling.gravity = {0, -9.81, 0};
    falling.step();
    check(
        std::abs(falling.particles[0].pos.y() + 0.0981) < 1e-12 &&
        std::abs(falling.particles[0].vel.y() + 0.981) < 1e-12,
        "unconstrained gravity integration"
    );

    auto solver = makePair();
    solver.particles[1].pos.setZero();
    solver.step();
    for (const auto& p : solver.particles)
        check(p.pos.allFinite() && p.vel.allFinite(), "coincident endpoints remain finite");

    for (auto& p : solver.particles) p.invMass = 0;
    const Eigen::Vector3d a = solver.particles[0].pos;
    const Eigen::Vector3d b = solver.particles[1].pos;
    solver.step();
    check(solver.particles[0].pos == a && solver.particles[1].pos == b, "all-fixed system");
}

void clothEvolution() {
    const auto fixture = makeClothFixture(17, 21, 0.18, 0.01);
    avbd::Solver solver;
    for (const auto& p : fixture.simulation.particles)
        solver.particles.push_back({p.pos, p.prevPos, Eigen::Vector3d::Zero(), p.invMass});
    for (const auto& c : fixture.simulation.constraints)
        solver.constraints.push_back({c.i, c.j, c.restLength, c.compliance});
    solver.dt = 1.0 / 60;
    solver.substeps = 10;

    check(
        fixture.triangles.size() == 640 && fixture.stretchCount == 996 &&
        solver.constraints.size() == 1920 && solver.constraints.size() - fixture.stretchCount == 924,
        "shared scene topology"
    );

    const Eigen::Vector3d left = solver.particles[0].pos;
    const Eigen::Vector3d right = solver.particles[20].pos;
    solver.particles[170].pos.z() = 0.15;
    double maxStretch = 0;

    for (int frame = 0; frame < 180; ++frame) {
        if (frame == 60) {
            solver.particles[170].pos += Eigen::Vector3d(0.1, 0.1, 0.2);
            solver.particles[170].invMass = 0;
        }
        if (frame == 70) solver.particles[170].invMass = 100;
        if (frame == 100) {
            solver.substeps = 2;
            for (size_t i = 0; i < fixture.stretchCount; ++i) solver.constraints[i].compliance = 1e-4;
        }
        const Eigen::Vector3d grab = solver.particles[170].pos;
        solver.step();
        if (frame >= 60 && frame < 70)
            check(solver.particles[170].pos == grab, "grabbed particle remains fixed");
        check(solver.particles[0].pos == left && solver.particles[20].pos == right, "cloth corner pins remain exact");
        for (const auto& p : solver.particles)
            check(
                p.pos.allFinite() && p.vel.allFinite() && p.pos.norm() < 100 && p.vel.norm() < 1e4,
                "bounded finite cloth evolution"
            );
        for (size_t i = 0; i < fixture.stretchCount; ++i) {
            const auto& c = solver.constraints[i];
            const double stretch = (solver.particles[c.i].pos - solver.particles[c.j].pos).norm() / c.restLength - 1;
            maxStretch = std::max(maxStretch, std::abs(stretch));
        }
    }
    std::printf("AVBD 180-frame edited/grabbed cloth peak relative stretch: %.6g\n", maxStretch);
}

void gradientChecks() {
    for (double compliance : {0.01, 0.0}) {
        auto solver = makePair(compliance);
        for (auto& c : solver.constraints) c.warmStart(solver.particles, solver.alpha, solver.gamma);
        if (compliance == 0) solver.constraints[0].lambda = 2;
        auto& p = solver.particles[1];
        p.pos = {0.4, 0.2, -0.3};
        p.inertialPos = {0.8, 0.1, 0.2};
        const double h = solver.dt / solver.substeps;
        Eigen::Vector3d gradient;
        Eigen::Matrix3d hessian;
        const double inertia = 1 / (p.invMass * h * h);
        solver.objectiveDerivatives(1, inertia, gradient, hessian);
        check(hessian.ldlt().isPositive(), "lumped Hessian is positive definite");
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d plus = p.pos;
            Eigen::Vector3d minus = p.pos;
            plus[axis] += 1e-6;
            minus[axis] -= 1e-6;
            const double derivative = (solver.objectiveValue(1, plus, inertia) - solver.objectiveValue(1, minus, inertia)) / 2e-6;
            check(std::abs(derivative - gradient[axis]) < 1e-5, "augmented objective gradient matches finite differences");
        }
    }
}

void pendulumChecks() {
    const Eigen::Vector3d anchor {0, 3, 0};
    const Eigen::Vector3d bob = anchor + Eigen::Vector3d(std::sin(0.65), -std::cos(0.65), 0);
    for (double compliance : {1e-6, 0.0}) {
        avbd::Solver solver;
        solver.particles = {
            {anchor, anchor, Eigen::Vector3d::Zero(), 0},
            {bob, bob, Eigen::Vector3d::Zero(), 1}
        };
        solver.constraints = {{0, 1, 1, compliance}};
        Eigen::Vector2d reference {0.65, 0};
        const auto rhs = [](const Eigen::Vector2d& state) -> Eigen::Vector2d {
            return {state.y(), -9.81 * std::sin(state.x())};
        };
        double linkError = 0;
        double angleError = 0;
        for (int frame = 0; frame < 600; ++frame) {
            for (int substep = 0; substep < 32; ++substep) {
                const double h = solver.dt / 32;
                const Eigen::Vector2d a = rhs(reference), b = rhs(reference + 0.5 * h * a);
                const Eigen::Vector2d c = rhs(reference + 0.5 * h * b), d = rhs(reference + h * c);
                reference += h / 6 * (a + 2 * b + 2 * c + d);
            }
            solver.step();
            const auto& p = solver.particles[1];
            const Eigen::Vector3d offset = p.pos - anchor;
            check(p.pos.allFinite() && p.vel.allFinite(), "pendulum remains finite");
            check(solver.particles[0].pos == anchor, "pendulum anchor stays fixed");
            linkError = std::max(linkError, std::abs(offset.norm() - 1));
            angleError = std::max(angleError, std::abs(std::atan2(offset.x(), -offset.y()) - reference.x()));
        }
        // Accuracy bounds for the default one-sweep, 10-substep budget.
        check(linkError < 0.02 && angleError < 0.1, "pendulum link and independent RK4 angle bounds");
        check(solver.constraints[0].penalty > 1000, "pendulum builds constraint penalty");
        check(compliance == 0 ? solver.constraints[0].lambda != 0 : solver.constraints[0].lambda == 0,
            "only hard constraints retain a multiplier");
        std::printf("AVBD pendulum compliance=%g: peak link error %.6g m, RK4 angle error %.6g rad\n",
            compliance, linkError, angleError);
    }
}

int main() {
    analyticSweep(0.01, 2, 0.2, -0.7);
    analyticSweep(0, 2, 0, 0);
    analyticSweep(0, 1, 0, 10);
    constraintState();
    gradientChecks();
    pendulumChecks();
    edgeCases();
    clothEvolution();
    if (failures) std::fprintf(stderr, "%d AVBD checks failed\n", failures);
    else std::puts("All AVBD checks passed");
    return failures ? 1 : 0;
}
