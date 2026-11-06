#include "pbd.hpp"

ParticleSolver::ParticleSolver(xpbd::Solver state)
    : stretchConstraints(std::move(state.constraints)), gravity(state.gravity),
      substeps(state.substeps) {
    for (const auto& p : state.particles) {
        particles.push_back({p.pos, p.prevPos, p.vel, p.invMass});
    }
    substeps = 4;
    for (auto& c : stretchConstraints) {
        c.compliance = 1e-6;
    }
    rebuildConnectivity();
}

int ParticleSolver::constraintCount() const {
    return int(stretchConstraints.size() + bendConstraints.size());
}

xpbd::DistanceConstraint& ParticleSolver::constraint(int e) {
    return e < int(stretchConstraints.size()) ? stretchConstraints[e]
        : bendConstraints[e - stretchConstraints.size()];
}

const xpbd::DistanceConstraint& ParticleSolver::constraint(int e) const {
    return e < int(stretchConstraints.size()) ? stretchConstraints[e]
        : bendConstraints[e - stretchConstraints.size()];
}

void ParticleSolver::rebuildConnectivity() {
    incident.assign(particles.size(), {});
    for (int e = 0; e < constraintCount(); ++e) {
        const auto& c = constraint(e);
        incident[c.i].push_back(e);
        incident[c.j].push_back(e);
    }
}

double ParticleSolver::substepSize(double dt) const {
    if (!(dt > 0) || !std::isfinite(dt) || substeps < 1 || iterations < 1) {
        throw std::runtime_error("Positive timestep, substeps, and iterations required");
    }
    return dt / substeps;
}

void ParticleSolver::predict(double h) {
    targets.resize(particles.size());
    for (size_t i = 0; i < particles.size(); ++i) {
        auto& p = particles[i];
        p.prevPos = p.pos;
        targets[i] = p.pos;
        if (p.invMass == 0) {
            continue;
        }
        targets[i] += h * p.vel + h * h * gravity;
        p.pos = targets[i];
    }
}

void ParticleSolver::updateVelocities(double h) {
    for (auto& p : particles) {
        if (p.invMass == 0) {
            p.vel.setZero();
        } else {
            p.vel = (p.pos - p.prevPos) / h;
        }
    }
}

double ParticleSolver::stiffness(const xpbd::DistanceConstraint& c) {
    if (!(c.compliance > 0) || !std::isfinite(c.compliance)) {
        throw std::runtime_error("This elastic solver requires finite positive compliance; "
            "use PBD/XPBD/AVBD for hard constraints");
    }
    return 1 / c.compliance;
}

double ParticleSolver::localEnergy(int i, const Eigen::Vector3d& x, double h) const {
    double energy = (x - targets[i]).squaredNorm() / (2 * particles[i].invMass * h * h);
    for (int e : incident[i]) {
        const auto& c = constraint(e);
        const int j = c.i == i ? c.j : c.i;
        const double C = (x - particles[j].pos).norm() - c.restLength;
        energy += 0.5 * stiffness(c) * C * C;
    }
    return energy;
}

void Pbd::project(vbd::Particle& a, vbd::Particle& b, double rest, double stiffness) {
    const Eigen::Vector3d d = b.pos - a.pos;
    const double length = d.norm(), w = a.invMass + b.invMass;
    if (length < 1e-12 || w == 0) return;
    const Eigen::Vector3d correction = stiffness * (length - rest) / (w * length) * d;
    a.pos += a.invMass * correction;
    b.pos -= b.invMass * correction;
}

void Pbd::step(double dt) {
    const double h = substepSize(dt);
    // Paper Sec. 3.3: convert substep stiffness to per-sweep stiffness.
    // This normalizes an isolated constraint, not a coupled cloth's convergence.
    const double stretch = 1 - std::pow(1 - stretchStiffness, 1.0 / iterations);
    const double bending = 1 - std::pow(1 - bendingStiffness, 1.0 / iterations);
    for (int substep = 0; substep < substeps; ++substep) {
        predict(h);
        for (int it = 0; it < iterations; ++it) {
            for (const auto& c : stretchConstraints)
                project(particles[c.i], particles[c.j], c.restLength, stretch);
            for (const auto& c : bendConstraints)
                project(particles[c.i], particles[c.j], c.restLength, bending);
        }
        updateVelocities(h);
    }
}
