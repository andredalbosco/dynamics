#include "core/xpbd.hpp"
#include "core/vbd.hpp"
#include "core/pd.hpp"
#include "core/avbd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

bool compareSolvers(double invMass, double initialVelocity) {
    const Eigen::Vector3d anchor {-1, 2.5, 0};
    const Eigen::Vector3d bob {0.5, 2.5, 0};
    constexpr double dt = 1.0 / 60;
    constexpr double stiffness = 100;
    constexpr int substeps = 10;
    const double h = dt / substeps;
    const double omega = std::sqrt(stiffness * invMass);

    xpbd::Solver xpbd;
    xpbd.particles = {{anchor, anchor, Eigen::Vector3d::Zero(), 0}, {bob, bob, {initialVelocity, 0, 0}, invMass}};
    xpbd.constraints = {{0, 1, 1, 1 / stiffness}};
    xpbd.substeps = substeps;
    xpbd.dt = dt;
    xpbd.gravity.setZero();

    vbd::Solver vbd;
    avbd::Solver avbd;
    for (const auto& p : xpbd.particles) {
        vbd.particles.push_back({p.pos, p.prevPos, p.vel, p.invMass});
        avbd.particles.push_back({p.pos, p.prevPos, p.vel, p.invMass});
    }
    vbd.energies = {{0, 1, 1, stiffness}};
    avbd.constraints = {{0, 1, 1, 1 / stiffness}};
    vbd.gravity.setZero();
    avbd.gravity.setZero();
    vbd.substeps = avbd.substeps = substeps;
    vbd.dt = avbd.dt = dt;

    pd::Solver pd;
    auto& ps = pd.particles;
    ps.resize(2);
    ps.pos.row(0) = anchor.transpose();
    ps.pos.row(1) = bob.transpose();
    ps.prevPos = ps.pos;
    ps.vel(1, 0) = initialVelocity;
    ps.invMass << 0, invMass;
    pd.energies = {{0, 1, 1, stiffness}};
    pd.gravity.setZero();
    pd.substeps = substeps;
    pd.dt = dt;
    pd.factorize();

    double positionError = 0;
    double velocityError = 0;
    double solverDifference = 0;
    double continuousError = 0;
    for (int frame = 1; frame <= 600; ++frame) {
        xpbd.step();
        vbd.step();
        pd.step();
        avbd.step();

        // Closed-form implicit-Euler oscillator, including its numerical damping.
        const int steps = frame * substeps;
        const double decay = std::pow(1 + h * h * omega * omega, -0.5 * steps);
        const double angle = steps * std::atan(h * omega);
        const double position = decay * (0.5 * std::cos(angle) + initialVelocity / omega * std::sin(angle));
        const double velocity = decay * (initialVelocity * std::cos(angle) - 0.5 * omega * std::sin(angle));
        for (double x : {xpbd.particles[1].pos.x(), vbd.particles[1].pos.x(), ps.pos(1, 0), avbd.particles[1].pos.x()}) {
            if (!std::isfinite(x)) return false;
            positionError = std::max(positionError, std::abs(x - position));
            solverDifference = std::max(solverDifference, std::abs(x - ps.pos(1, 0)));
        }
        for (double v : {xpbd.particles[1].vel.x(),
                vbd.particles[1].vel.x(), ps.vel(1, 0), avbd.particles[1].vel.x()}) {
            if (!std::isfinite(v)) return false;
            velocityError = std::max(velocityError, std::abs(v - velocity));
        }
        if (xpbd.particles[0].pos != anchor || vbd.particles[0].pos != anchor ||
            ps.pos.row(0) != anchor.transpose() || avbd.particles[0].pos != anchor) return false;
        const double time = frame * dt;
        continuousError = std::max(continuousError, std::abs(ps.pos(1, 0) -
            (0.5 * std::cos(omega * time) + initialVelocity / omega * std::sin(omega * time))));
    }
    std::printf("Spring invMass=%g v0=%g: solver difference %.3g m, discrete reference error %.3g m / %.3g m/s, continuous error %.3g m\n",
        invMass, initialVelocity, solverDifference, positionError, velocityError, continuousError);
    return solverDifference < 1e-8 && positionError < 1e-8 && velocityError < 1e-7;
}

int main() {
    for (double invMass : {1.0, 0.5}) {
        for (double velocity : {0.0, 0.7}) {
            if (!compareSolvers(invMass, velocity)) return 1;
        }
    }
    return 0;
}
