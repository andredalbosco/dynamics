#include "vbd.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>

namespace vbd {

constexpr double epsilon = 1e-12;
constexpr double hessianFloor = 1e-6;
constexpr int lineSearchSteps = 24;

Eigen::Matrix3d projectPSD(
    const Eigen::Matrix3d& matrix,
    double minEigenvalue
) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(matrix);
    if (eigen.info() != Eigen::Success) throw std::runtime_error("VBD eigensolve failed");
    return eigen.eigenvectors()
        * eigen.eigenvalues().cwiseMax(minEigenvalue).asDiagonal()
        * eigen.eigenvectors().transpose();
}

int SpringEnergy::neighbor(int vertex) const {
    if (vertex == i) return j;
    if (vertex == j) return i;
    return -1;
}

double springEnergyValue(
    const Eigen::Vector3d& delta,
    double restLength,
    double stiffness
) {
    const double extension = delta.norm() - restLength;
    return 0.5 * stiffness * extension * extension;
}

void springEnergyDerivatives(
    const Eigen::Vector3d& delta,
    double restLength,
    double stiffness,
    Eigen::Vector3d& gradient,
    Eigen::Matrix3d& hessian
) {
    const double length = delta.norm();
    if (length < epsilon) return;

    const Eigen::Vector3d n = delta / length;
    const Eigen::Matrix3d nn = n * n.transpose();
    const double force = stiffness * (length - restLength);

    gradient += force * n;
    hessian += stiffness * nn;
    hessian += force / length * (Eigen::Matrix3d::Identity() - nn);
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

void Solver::energyDerivatives(
    int i,
    Eigen::Vector3d& gradient,
    Eigen::Matrix3d& hessian
) const {
    auto& p = particles[i];

    for (const auto& e : energies) {
        const int neighbor = e.neighbor(i);
        if (neighbor < 0) continue;
        springEnergyDerivatives(
            p.pos - particles[neighbor].pos,
            e.restLength,
            e.stiffness,
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

    for (const auto& e : energies) {
        const int neighbor = e.neighbor(i);
        if (neighbor < 0) continue;
        objective += springEnergyValue(
            position - particles[neighbor].pos,
            e.restLength,
            e.stiffness
        );
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
    energyDerivatives(i, gradient, hessian);
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
    hessian = projectPSD(hessian, hessianFloor * inertia);

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

        // Vertex-wise optimization
        for (int i = 0; i < int(particles.size()); ++i) {
            if (particles[i].invMass == 0) continue;
            optimizeVertex(i, h);
        }

        updateVelocities(h);
    }
}

} // namespace vbd
