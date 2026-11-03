#include "avbd.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>

namespace avbd {

constexpr double epsilon = 1e-12;
constexpr int lineSearchSteps = 24;
constexpr double startPenalty = 1000;
constexpr double maxPenalty = 1e12;

int DistanceConstraint::neighbor(int vertex) const {
    if (vertex == i) return j;
    if (vertex == j) return i;
    return -1;
}

double DistanceConstraint::residual(double length, double alpha) const {
    return length - restLength - (compliance == 0 ? alpha * initialResidual : 0);
}

void DistanceConstraint::warmStart(
    const std::vector<Particle>& particles,
    double alpha,
    double gamma
) {
    initialResidual = (particles[i].prevPos - particles[j].prevPos).norm() - restLength;
    const double cap = compliance == 0 ? maxPenalty : 1 / compliance;
    penalty = std::min(cap, std::max(startPenalty, gamma * penalty));
    lambda = compliance == 0 ? alpha * gamma * lambda : 0;
}

void DistanceConstraint::update(
    const std::vector<Particle>& particles,
    double alpha,
    double beta
) {
    const double r = residual((particles[i].pos - particles[j].pos).norm(), alpha);
    if (compliance == 0) lambda += penalty * r;
    penalty = std::min(compliance == 0 ? maxPenalty : 1 / compliance,
        penalty + beta * std::abs(r));
}

double augmentedEnergyValue(double r, double penalty, double lambda) {
    return 0.5 * penalty * r * r + lambda * r;
}

void augmentedEnergyDerivatives(
    const Eigen::Vector3d& delta,
    double r,
    double penalty,
    double lambda,
    Eigen::Vector3d& gradient,
    Eigen::Matrix3d& hessian
) {
    const double length = delta.norm();
    if (length < epsilon) return;

    const Eigen::Vector3d n = delta / length;
    const Eigen::Matrix3d nn = n * n.transpose();
    const double force = penalty * r + lambda;
    gradient += force * n;
    hessian += penalty * nn;

    // Curvature lumping
    const Eigen::Matrix3d geometric = force / length * (Eigen::Matrix3d::Identity() - nn);
    for (int axis = 0; axis < 3; ++axis) {
        hessian(axis, axis) += geometric.col(axis).norm();
    }
}

void Solver::predictPositions(double h) {
    for (auto& p : particles) {
        p.prevPos = p.pos;
        if (p.invMass == 0) {
            p.inertialPos = p.pos;
            continue;
        }

        p.inertialPos = p.pos + h * p.vel + h * h * gravity;
        p.pos = p.inertialPos;
    }
}

void Solver::inertialDerivatives(
    int i,
    double inertia,
    Eigen::Vector3d& gradient,
    Eigen::Matrix3d& hessian
) const {
    auto& p = particles[i];
    gradient += inertia * (p.pos - p.inertialPos);
    hessian += inertia * Eigen::Matrix3d::Identity();
}

void Solver::constraintDerivatives(
    int i,
    Eigen::Vector3d& gradient,
    Eigen::Matrix3d& hessian
) const {
    auto& p = particles[i];

    for (const auto& c : constraints) {
        const int neighbor = c.neighbor(i);
        if (neighbor < 0) continue;
        const Eigen::Vector3d delta = p.pos - particles[neighbor].pos;
        const double r = c.residual(delta.norm(), alpha);
        augmentedEnergyDerivatives(
            delta,
            r,
            c.penalty,
            c.lambda,
            gradient,
            hessian
        );
    }
}

double Solver::objectiveValue(
    int i,
    const Eigen::Vector3d& position,
    double inertia
) const {
    auto& p = particles[i];
    double objective = 0.5 * inertia * (position - p.inertialPos).squaredNorm();

    for (const auto& c : constraints) {
        const int neighbor = c.neighbor(i);
        if (neighbor < 0) continue;
        const double r = c.residual((position - particles[neighbor].pos).norm(), alpha);
        objective += augmentedEnergyValue(r, c.penalty, c.lambda);
    }

    return objective;
}

void Solver::objectiveDerivatives(
    int i,
    double inertia,
    Eigen::Vector3d& gradient,
    Eigen::Matrix3d& hessian
) const {
    gradient.setZero();
    hessian.setZero();
    inertialDerivatives(i, inertia, gradient, hessian);
    constraintDerivatives(i, gradient, hessian);
}

Eigen::Vector3d Solver::lineSearch(int i, Eigen::Vector3d dx, double inertia) const {
    auto& p = particles[i];
    const double objective = objectiveValue(i, p.pos, inertia);

    for (int search = 0; search < lineSearchSteps; ++search, dx *= 0.5) {
        const Eigen::Vector3d nextPos = p.pos + dx;
        if (nextPos.allFinite() && objectiveValue(i, nextPos, inertia) <= objective) return dx;
    }
    return Eigen::Vector3d::Zero();
}

void Solver::optimizeVertex(int i, double h) {
    auto& p = particles[i];
    const double inertia = 1 / (p.invMass * h * h);

    Eigen::Vector3d gradient;
    Eigen::Matrix3d hessian;

    objectiveDerivatives(i, inertia, gradient, hessian);
    const Eigen::Vector3d dx = -hessian.ldlt().solve(gradient);
    p.pos += lineSearch(i, dx, inertia);
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

        for (auto& c : constraints) {
            c.warmStart(particles, alpha, gamma);
        }

        // Vertex-wise optimization
        for (int i = 0; i < int(particles.size()); ++i) {
            if (particles[i].invMass == 0) continue;
            optimizeVertex(i, h);
        }

        for (auto& c : constraints) {
            c.update(particles, alpha, beta);
        }

        updateVelocities(h);
    }
}

} // namespace avbd
