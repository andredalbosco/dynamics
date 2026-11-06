#include "bw.hpp"

#include <Eigen/SparseLU>
#include <stdexcept>

namespace bw {

constexpr double epsilon = 1e-12;

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

void Solver::energyDerivatives(
    Eigen::VectorXd& gradient,
    Eigen::SparseMatrix<double>& hessian
) const {
    std::vector<Eigen::Triplet<double>> entries;

    // grad U(x), Hessian of U(x)
    for (const auto& e : energies) {
        Eigen::Vector3d springGradient = Eigen::Vector3d::Zero();
        Eigen::Matrix3d springHessian = Eigen::Matrix3d::Zero();
        springEnergyDerivatives(
            particles[e.i].pos - particles[e.j].pos,
            e.restLength,
            e.stiffness,
            springGradient,
            springHessian
        );
        gradient.segment<3>(3 * e.i) += springGradient;
        gradient.segment<3>(3 * e.j) -= springGradient;

        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                const double value = springHessian(a, b);
                entries.emplace_back(3 * e.i + a, 3 * e.i + b, value);
                entries.emplace_back(3 * e.j + a, 3 * e.j + b, value);
                entries.emplace_back(3 * e.i + a, 3 * e.j + b, -value);
                entries.emplace_back(3 * e.j + a, 3 * e.i + b, -value);
            }
        }
    }

    Eigen::SparseMatrix<double> springHessian(hessian.rows(), hessian.cols());
    springHessian.setFromTriplets(entries.begin(), entries.end());
    hessian += springHessian;
}

void Solver::solveVelocities(double h) {
    const int count = 3 * int(particles.size());
    if (count == 0) return;

    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(count);
    Eigen::SparseMatrix<double> hessian(count, count);
    energyDerivatives(gradient, hessian);

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(count);
    std::vector<Eigen::Triplet<double>> entries;

    // (M + h2 * Hessian U) * nextVel = M * vel + h * (M * gravity - grad U)
    for (int i = 0; i < int(particles.size()); ++i) {
        auto& p = particles[i];
        const double mass = p.invMass == 0 ? 1 : 1 / p.invMass;
        for (int axis = 0; axis < 3; ++axis)
            entries.emplace_back(3 * i + axis, 3 * i + axis, mass);
        if (p.invMass == 0) continue;
        rhs.segment<3>(3 * i) = mass * (p.vel + h * gravity) - h * gradient.segment<3>(3 * i);
    }

    // Fixed velocities are zero: remove their rows and columns from the spring blocks.
    for (int column = 0; column < hessian.outerSize(); ++column) {
        for (Eigen::SparseMatrix<double>::InnerIterator entry(hessian, column); entry; ++entry) {
            if (particles[entry.row() / 3].invMass == 0 || particles[entry.col() / 3].invMass == 0) continue;
            entries.emplace_back(entry.row(), entry.col(), h * h * entry.value());
        }
    }

    Eigen::SparseMatrix<double> matrix(count, count);
    matrix.setFromTriplets(entries.begin(), entries.end());

    // Exact spring curvature can make the system indefinite under compression.
    Eigen::SparseLU<Eigen::SparseMatrix<double>> factor;
    factor.compute(matrix);
    if (factor.info() != Eigen::Success) throw std::runtime_error("BW factorization failed");
    const Eigen::VectorXd velocity = factor.solve(rhs);
    if (factor.info() != Eigen::Success || !velocity.allFinite())
        throw std::runtime_error("BW velocity solve failed");

    for (int i = 0; i < int(particles.size()); ++i)
        particles[i].vel = velocity.segment<3>(3 * i);
}

void Solver::updatePositions(double h) {
    for (auto& p : particles) {
        p.prevPos = p.pos;
        if (p.invMass == 0) continue;
        p.pos += h * p.vel;
    }
}

void Solver::step() {
    const double h = dt / substeps;

    for (int substep = 0; substep < substeps; ++substep) {
        solveVelocities(h);
        updatePositions(h);
    }
}

} // namespace bw
