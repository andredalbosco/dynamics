#include "xpbd.hpp"

namespace xpbd {

constexpr double epsilon = 1e-12;

void DistanceConstraint::project(std::vector<Particle>& particles, double h) {
    lambda = 0;

    auto& pi = particles[i];
    auto& pj = particles[j];

    const Eigen::Vector3d delta = pj.pos - pi.pos;
    const double length = delta.norm();
    const double w = pi.invMass + pj.invMass;

    if (length < epsilon || w == 0) return;

    const Eigen::Vector3d n = delta / length;
    const double alpha_tilde = compliance / (h * h);
    const double dlambda = -(length - restLength) / (w + alpha_tilde);

    pi.pos -= pi.invMass * dlambda * n;
    pj.pos += pj.invMass * dlambda * n;
    lambda = dlambda;
}

void Solver::predictPositions(double h) {
    for (auto& p : particles) {
        p.prevPos = p.pos;
        if (p.invMass == 0) continue;

        p.pos += h * p.vel + h * h * gravity;
    }
}

void Solver::updateVelocities(double h) {
    for (auto& p : particles) {
        if (p.invMass == 0) p.vel.setZero();
        else p.vel = (p.pos - p.prevPos) / h;
    }
}

void Solver::step() {
    const double h = dt / substeps;

    for (int substep = 0; substep < substeps; ++substep) {
        predictPositions(h);
        for (auto& c : constraints) c.project(particles, h);
        updateVelocities(h);
    }
}

} // namespace xpbd
